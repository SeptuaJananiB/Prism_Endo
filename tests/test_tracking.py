import numpy as np
import pandas as pd
import pytest

from src.tracking.dlc_io import DLCPredictions, read_dlc_table, table_to_keypoints
from src.tracking.keypoints import Keypoint, KeypointSpec, gate
from src.tracking.mask_tip import estimate_tip, field_of_view_mask
from src.tracking.sources import DLCKeypointSource, MaskTipKeypointSource


def shaft_mask(h=200, w=300):
    """Instrument entering from the left edge, tip at x=180."""
    m = np.zeros((h, w), np.uint8)
    m[95:106, 0:181] = 1
    return m


def test_tip_is_far_end_from_fov_entry():
    est = estimate_tip(shaft_mask(), None, min_area=100)
    assert est["method"] == "farthest_from_fov_entry"
    assert est["x"] == pytest.approx(180, abs=2)
    assert est["y"] == pytest.approx(100, abs=4)
    assert 0 < est["confidence"] <= 1


def test_tip_with_fov_border():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, 40:260] = 150  # illuminated field with black side borders
    fov = field_of_view_mask(img)
    m = np.zeros((200, 300), np.uint8)
    m[95:106, 40:200] = 1  # enters at the FOV edge x=40, not the image edge
    est = estimate_tip(m, fov, min_area=100)
    assert est["x"] == pytest.approx(199, abs=2)
    assert est["method"] == "farthest_from_fov_entry"


def test_floating_mask_uses_principal_axis_with_lower_confidence():
    m = np.zeros((200, 300), np.uint8)
    m[95:106, 100:200] = 1
    est = estimate_tip(m, None, min_area=100)
    assert est["method"] == "principal_axis_end"
    full = estimate_tip(shaft_mask(), None, min_area=100)
    assert est["confidence"] < full["confidence"] or full["confidence"] == 1.0


def test_empty_or_tiny_mask_returns_none():
    assert estimate_tip(np.zeros((50, 50), np.uint8)) is None
    m = np.zeros((50, 50), np.uint8)
    m[0:3, 0:3] = 1
    assert estimate_tip(m, min_area=100) is None


def test_gate_statuses():
    assert gate(Keypoint("a", 1, 2, 0.9), 0.5).status == "ok"
    assert gate(Keypoint("a", 1, 2, 0.1), 0.5).status == "low_confidence"
    assert gate(Keypoint("a", None, None, 0.9), 0.5).status == "missing"
    assert gate(Keypoint("a", float("nan"), 2, 0.9), 0.5).status == "missing"
    assert not gate(Keypoint("a", 1, 2, 0.1), 0.5).valid


def test_mask_tip_source_per_instrument_class():
    labels = np.zeros((200, 300), np.uint8)
    labels[95:106, 0:181] = 2  # l_hook
    src = MaskTipKeypointSource([KeypointSpec("lhook_tip", "l_hook"), KeypointSpec("grasper_tip", "grasper")],
                                ["background", "grasper", "l_hook"], min_area=100)
    img = np.full((200, 300, 3), 120, np.uint8)
    kps = {k.name: k for k in src(img, 0, labels)}
    assert kps["lhook_tip"].valid and kps["lhook_tip"].x == pytest.approx(180, abs=2)
    assert kps["grasper_tip"].status == "missing"  # no grasper visible -> no invented point
    assert kps["lhook_tip"].source == "mask_heuristic"
    with pytest.raises(ValueError):
        MaskTipKeypointSource([KeypointSpec("x", "scissors")], ["background", "grasper"])


def write_dlc_csv(path, rows, bodyparts=("lhook_tip", "grasper_tip"), scorer="DLC_resnet50_test"):
    cols = pd.MultiIndex.from_product([[scorer], list(bodyparts), ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path)
    return df


def test_read_dlc_prediction_csv(tmp_path):
    p = tmp_path / "clipDLC_resnet50_test.csv"
    write_dlc_csv(p, [[10, 20, 0.95, 30, 40, 0.2], [11, 21, 0.9, np.nan, np.nan, 0.0]])
    df = read_dlc_table(p)
    kp = table_to_keypoints(df)
    assert set(kp) == {"lhook_tip", "grasper_tip"}
    assert kp["lhook_tip"]["x"][1] == 11
    pred = DLCPredictions(p)
    assert pred.get(0, "lhook_tip") == (10.0, 20.0, 0.95)
    assert pred.get(99, "lhook_tip") is None
    with pytest.raises(KeyError):
        pred.get(0, "nonexistent")


def test_dlc_source_gates_low_likelihood(tmp_path):
    p = tmp_path / "clipDLC_resnet50_test.csv"
    write_dlc_csv(p, [[10, 20, 0.95, 30, 40, 0.2]])
    src = DLCKeypointSource([KeypointSpec("lhook_tip", "l_hook", "lhook_tip"),
                             KeypointSpec("grasper_tip", "grasper", "grasper_tip")], p, min_confidence=0.6)
    kps = {k.name: k for k in src(None, 0)}
    assert kps["lhook_tip"].status == "ok" and kps["lhook_tip"].source == "dlc"
    assert kps["grasper_tip"].status == "low_confidence"
    assert {k.status for k in src(None, 5)} == {"missing"}  # frame not in file


def test_dlc_source_rejects_missing_bodypart(tmp_path):
    p = tmp_path / "clipDLC_resnet50_test.csv"
    write_dlc_csv(p, [[10, 20, 0.95]], bodyparts=("lhook_tip",))
    with pytest.raises(ValueError):
        DLCKeypointSource([KeypointSpec("grasper_tip", "grasper", "grasper_tip")], p)


def test_read_dlc_label_table_with_path_index(tmp_path):
    # CollectedData format: no likelihood; unlabelled points are NaN.
    cols = pd.MultiIndex.from_product([["annotator"], ["lhook_tip"], ["x", "y"]],
                                      names=["scorer", "bodyparts", "coords"])
    idx = ["labeled-data/clip/img0003.png", "labeled-data/clip/img0010.png"]
    df = pd.DataFrame([[5.0, 6.0], [np.nan, np.nan]], index=idx, columns=cols)
    p = tmp_path / "CollectedData_annotator.csv"
    df.to_csv(p)
    pred = DLCPredictions(p)
    assert pred.get(3, "lhook_tip") == (5.0, 6.0, 1.0)
    x, y, lik = pred.get(10, "lhook_tip")
    assert np.isnan(x) and lik == 0.0  # unlabelled is not ground truth
