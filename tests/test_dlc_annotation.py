import numpy as np
import pandas as pd
import pytest
import yaml

from src.tracking.annotation_prep import FrameInfo, dlc_image_name, farthest_point_sample, select_frames
from src.tracking.dlc_workflow import label_status, require_labels


def fi(clip, i, g=0, h=0, inst=0, feat=None):
    return FrameInfo(clip, i, 1000 + i, None, g, h, inst, np.array(feat if feat is not None else [i, g > 0, h > 0], float))


def test_dlc_image_name_matches_dlc_padding():
    assert dlc_image_name(5, 80) == "img05.png"
    assert dlc_image_name(5, 2000) == "img0005.png"


def test_farthest_point_sample_is_diverse_and_spaced():
    items = [fi("c", i, g=500, feat=[i]) for i in range(20)]
    picked = farthest_point_sample(items, 4, min_gap=2)
    idx = [f.index for f in picked]
    assert len(idx) == 4 and max(idx) - min(idx) >= 15          # spread over the clip
    assert all(b - a >= 2 for a, b in zip(idx, idx[1:]))


def test_select_frames_skips_instrument_free_frames_and_respects_budget():
    clips = {"a": [fi("a", i, g=500) for i in range(30)], "b": [fi("b", i) for i in range(30)],
             "c": [fi("c", i, h=400) for i in range(30)]}
    picked = select_frames(clips, 20)
    assert len(picked) <= 20 and {f.clip for f in picked} == {"a", "c"}
    assert all(f.grasper_px or f.lhook_px for f in picked)


def make_project(tmp_path, labels):
    proj = tmp_path / "proj"
    (proj / "labeled-data" / "vid").mkdir(parents=True)
    bps = ["lhook_tip", "grasper_tip", "grasper_tip_2"]
    (proj / "config.yaml").write_text(yaml.safe_dump({"bodyparts": bps}))
    n = len(labels)
    for i in range(n):
        (proj / "labeled-data" / "vid" / f"img{i:03d}.png").write_bytes(b"")
    if n:
        cols = pd.MultiIndex.from_product([["human"], bps, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
        idx = pd.MultiIndex.from_tuples([("labeled-data", "vid", f"img{i:03d}.png") for i in range(n)])
        pd.DataFrame(labels, index=idx, columns=cols).to_csv(proj / "labeled-data" / "vid" / "CollectedData_human.csv")
    return proj / "config.yaml"


def test_gate_refuses_without_labels(tmp_path):
    cfg = make_project(tmp_path, [])
    assert label_status(cfg)["frames_labelled"] == 0
    with pytest.raises(RuntimeError, match="Not enough human labels"):
        require_labels(cfg)


def test_gate_allows_enough_labels_and_ignores_unlabelled(tmp_path):
    # Synthetic TEST FIXTURE coordinates (not real annotations). grasper_tip_2 mostly unlabelled (NaN).
    nan = float("nan")
    rows = [[10, 20, 30, 40, nan, nan] for _ in range(120)] + [[nan] * 6 for _ in range(5)]
    cfg = make_project(tmp_path, rows)
    st = require_labels(cfg, min_frames=100, min_per_bodypart=30)
    assert st["frames_labelled"] == 120 and st["per_bodypart"]["grasper_tip_2"] == 0
    assert any("grasper_tip_2" in w for w in st["warnings"])
    with pytest.raises(RuntimeError):
        require_labels(cfg, min_frames=200)
