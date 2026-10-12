"""Multi-video dataset handling: list-of-roots discovery, clip/video grouping,
leakage checks and the video-level split proposal (scripts/make_splits.py).
All data is synthetic (tiny PNGs in tmp dirs)."""

import importlib.util
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.data.segmentation_data import (Sample, discover_from_config, discover_samples, split_samples,
                                        videos_in_several_splits)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_splits", ROOT / "scripts" / "make_splits.py")
make_splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_splits)

TARGETS = [("grasper", [31]), ("l_hook", [32]), ("liver", [21]), ("cystic_duct", [25])]


def make_clip(folder: Path, start: int, values=(21,), n: int = 3) -> Path:
    """One CholecSeg8k-style clip folder with n frames; the label map holds
    each raw value in its own 8x8 block on a 50 (border) background."""
    folder.mkdir(parents=True)
    for i in range(start, start + n):
        cv2.imwrite(str(folder / f"frame_{i}_endo.png"), np.zeros((16, 32, 3), np.uint8))
        lab = np.full((16, 32, 3), 50, np.uint8)
        for k, v in enumerate(values):
            lab[:8, 8 * k:8 * (k + 1)] = v
        cv2.imwrite(str(folder / f"frame_{i}_endo_watershed_mask.png"), lab)
        cv2.imwrite(str(folder / f"frame_{i}_endo_mask.png"), np.zeros((16, 32), np.uint8))  # sparse; unused
    return folder


@pytest.fixture
def multi_root(tmp_path):
    """Existing-style video folder (root = video01) plus a CholecSeg8k download
    root holding two more videos."""
    v1 = tmp_path / "video01"
    make_clip(v1 / "video01_00080", 80)
    make_clip(v1 / "video01_00160", 160)
    dl = tmp_path / "data" / "cholecseg8k"
    make_clip(dl / "video09" / "video09_00100", 100)
    make_clip(dl / "video12" / "video12_15750", 15750)
    return v1, dl


def test_sample_keeps_positional_api_and_derives_video():
    s = Sample("a.png", "a_m.png", "video05_00080")
    assert s.group == "video05_00080" and s.video == "video05"
    assert Sample("a", "b", "g", "video07").video == "video07"


def test_list_of_roots_discovery(multi_root):
    v1, dl = multi_root
    single = discover_samples(v1)
    assert len(single) == 6 and {s.group for s in single} == {"video01_00080", "video01_00160"}
    assert {s.video for s in single} == {"video01"}
    both = discover_samples([v1, dl])
    assert len(both) == 12
    assert {s.group for s in both} == {"video01_00080", "video01_00160", "video09_00100", "video12_15750"}
    assert {s.video for s in both} == {"video01", "video09", "video12"}
    assert [s.image for s in both[:6]] == [s.image for s in single]   # roots scanned in the given order
    cfg = {"root": [str(v1), str(dl)], "layout": "cholecseg8k"}
    assert [s.image for s in discover_from_config(cfg)] == [s.image for s in both]
    assert discover_from_config({"root": str(v1)})[0].group == "video01_00080"


def test_group_by_video(multi_root):
    samples = discover_samples(list(multi_root), group_by="video")
    assert {s.group for s in samples} == {"video01", "video09", "video12"}
    assert all(s.group == s.video for s in samples)
    with pytest.raises(ValueError, match="group_by"):
        discover_samples(list(multi_root), group_by="patient")


def test_video_in_two_splits_is_an_error_with_group_by_video(multi_root):
    samples = discover_samples(list(multi_root), group_by="video")
    with pytest.raises(ValueError, match="leakage"):
        split_samples(samples, {"test": ["video09"], "train": ["video01", "video09"]})
    with pytest.raises(ValueError, match="unknown group"):   # clip ids are not video ids
        split_samples(samples, {"test": ["video09_00100"], "train": "rest"})
    sp = split_samples(samples, {"test": ["video12"], "val": ["video09"], "train": "rest"})
    assert {s.video for s in sp["train"]} == {"video01"} and not videos_in_several_splits(sp)


def test_within_patient_warning_with_group_by_clip(multi_root, caplog):
    samples = discover_samples(list(multi_root))   # group_by=clip (default)
    caplog.set_level(logging.WARNING, logger="prism.data")
    sp = split_samples(samples, {"test": ["video01_00080"], "train": "rest"})   # current design: allowed
    assert videos_in_several_splits(sp) == {"video01": ["test", "train"]}
    assert "Within-patient split" in caplog.text and "video01" in caplog.text
    caplog.clear()
    split_samples(samples, {"test": ["video12_15750"], "val": ["video09_00100"], "train": "rest"})
    assert "Within-patient" not in caplog.text   # clip lists that keep videos disjoint: no warning


def test_duplicate_and_overlapping_roots(multi_root, tmp_path, caplog):
    v1, dl = multi_root
    caplog.set_level(logging.WARNING, logger="prism.data")
    # overlapping roots: the same files are counted once
    assert len(discover_samples([dl, dl / "video09"])) == 6
    assert "overlap" in caplog.text
    # the same clip in two different folders (e.g. video01 also inside the download) is an error
    make_clip(dl / "video01" / "video01_00080", 80)
    with pytest.raises(ValueError, match="two folders"):
        discover_samples([v1, dl])


def test_indexed_png_list_of_roots(tmp_path):
    for r, names in (("a", ["case1_0001", "case1_0002"]), ("b", ["case2_0001"])):
        (tmp_path / r / "images").mkdir(parents=True)
        (tmp_path / r / "masks").mkdir(parents=True)
        for n in names:
            cv2.imwrite(str(tmp_path / r / "images" / f"{n}.png"), np.zeros((4, 4, 3), np.uint8))
            cv2.imwrite(str(tmp_path / r / "masks" / f"{n}.png"), np.ones((4, 4), np.uint8))
    s = discover_samples([tmp_path / "a", tmp_path / "b"], layout="indexed_png", group_by="video")
    assert [x.group for x in s] == ["case1", "case1", "case2"] and [x.video for x in s] == ["case1"] * 2 + ["case2"]


def four_video_tree(tmp_path) -> Path:
    root = tmp_path / "cholecseg8k"
    make_clip(root / "video01" / "video01_00000", 0, (31, 21))         # grasper, liver
    make_clip(root / "video02" / "video02_00000", 0, (31, 25))         # grasper, cystic duct (only here)
    make_clip(root / "video03" / "video03_00000", 0, (32, 21))         # L-hook, liver
    make_clip(root / "video04" / "video04_00000", 0, (31, 32, 21))     # grasper, L-hook, liver
    make_clip(root / "video04" / "video04_00500", 500, (31, 32, 21))
    return root


def test_split_sizes():
    assert make_splits.split_sizes(2) == (1, 0, 1)
    assert make_splits.split_sizes(3) == (1, 1, 1)
    assert make_splits.split_sizes(4) == (2, 1, 1)
    assert make_splits.split_sizes(17) == (11, 3, 3)
    with pytest.raises(ValueError):
        make_splits.split_sizes(1)


def test_target_classes_drop_merged_instrument():
    inst = [{"name": "background", "values": []}, {"name": "grasper", "values": [31]},
            {"name": "l_hook", "values": [32]}]
    tissue = [{"name": "background", "values": [50]}, {"name": "liver", "values": [21]},
              {"name": "instrument", "values": [31, 32]}]
    assert make_splits.target_classes([inst, tissue]) == [("grasper", [31]), ("l_hook", [32]), ("liver", [21])]


def test_video_split_proposal_on_four_videos(tmp_path):
    root = four_video_tree(tmp_path)
    samples = discover_samples(root)
    stats = make_splits.per_video(make_splits.scan(samples, TARGETS, every=1, min_area=10))
    assert stats["video04"]["frames"] == 6 and len(stats["video04"]["clips"]) == 2
    prop = make_splits.propose_video_split(stats, seed=0)
    sp = prop["splits"]
    # cystic duct exists only in video02 -> train; best test coverage is video04 (3 classes)
    assert sp == {"train": ["video02", "video03"], "val": ["video01"], "test": ["video04"]}
    assert prop["score"][:3] == (4, 3, 2)
    for seed in (1, 2, 3):   # the seed only breaks ties; this optimum is unique
        assert make_splits.propose_video_split(stats, seed=seed)["splits"] == sp


def test_make_splits_cli_writes_usable_video_split(tmp_path, capsys, caplog):
    root = four_video_tree(tmp_path)
    out = tmp_path / "splits.yaml"
    rc = make_splits.main(["--configs", str(ROOT / "configs/instrument_seg.yaml"),
                           str(ROOT / "configs/tissue_seg.yaml"), "--every", "1", "--min-area", "10",
                           "--out", str(out), f"data.root=[{root.as_posix()}]"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "Proposed VIDEO-level split" in text and "closest available proxy" in text
    snippet = yaml.safe_load(out.read_text(encoding="utf-8"))["data"]
    assert snippet["group_by"] == "video" and snippet["root"] == [root.as_posix()]
    assert sorted(sum(snippet["splits"].values(), [])) == ["video01", "video02", "video03", "video04"]
    assert "video02" in snippet["splits"]["train"]
    caplog.set_level(logging.WARNING, logger="prism.data")
    sp = split_samples(discover_from_config(snippet), snippet["splits"])   # the snippet is directly usable
    assert not videos_in_several_splits(sp) and "Within-patient" not in caplog.text


def test_make_splits_refuses_single_video(tmp_path, capsys):
    v1 = tmp_path / "video01"
    make_clip(v1 / "video01_00080", 80, (31, 21))
    make_clip(v1 / "video01_00160", 160, (32, 21))
    out = tmp_path / "splits.yaml"
    rc = make_splits.main(["--configs", str(ROOT / "configs/instrument_seg.yaml"), "--every", "1",
                           "--out", str(out), f"data.root={v1.as_posix()}",
                           "data.splits={test: [video01_00160], train: rest}"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "REFUSED" in text and "ONE video (video01)" in text and "WITHIN-PATIENT ONLY" in text
    assert "test :  1 clips" in text and "No file written." in text
    assert not out.exists()
