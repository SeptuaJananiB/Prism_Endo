import numpy as np
import pytest

from src.data.segmentation_data import IGNORE_INDEX, Sample, build_lut, split_samples
from src.segmentation.metrics import ConfusionMatrix, binary_dice_iou


def test_lut_maps_values_and_ignores():
    classes = [{"name": "background", "values": []}, {"name": "grasper", "values": [31]},
               {"name": "l_hook", "values": [32]}]
    lut = build_lut(classes, ignore_values=[255, 0])
    raw = np.array([[31, 32, 21, 255, 0, 50]], np.uint8)
    assert lut[raw].tolist() == [[1, 2, 0, IGNORE_INDEX, IGNORE_INDEX, 0]]


def test_split_by_group_and_rest():
    samples = [Sample(f"{g}_{i}.png", f"{g}_{i}_m.png", g) for g in ("a", "b", "c", "d") for i in range(3)]
    sp = split_samples(samples, {"test": ["a"], "val": ["b"], "train": "rest"})
    assert {s.group for s in sp["train"]} == {"c", "d"}
    assert len(sp["test"]) == 3 and len(sp["val"]) == 3
    groups = [{s.group for s in v} for v in sp.values()]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2])


def test_split_rejects_leakage_and_unknown_groups():
    samples = [Sample("x", "y", "a"), Sample("x", "y", "b")]
    with pytest.raises(ValueError, match="leakage"):
        split_samples(samples, {"train": ["a"], "test": ["a"]})
    with pytest.raises(ValueError, match="unknown"):
        split_samples(samples, {"test": ["zzz"]})


def test_confusion_matrix_dice_iou():
    gt = np.array([[0, 1, 1, 2], [0, 1, 255, 2]])
    pred = np.array([[0, 1, 0, 2], [0, 1, 1, 0]])
    cm = ConfusionMatrix(3)
    cm.update(pred, gt)
    pc = cm.per_class()
    # class 1: tp=2, fp=0 (ignored pixel excluded), fn=1 -> dice 4/5
    assert pc["dice"][1] == pytest.approx(0.8)
    assert pc["iou"][1] == pytest.approx(2 / 3)
    # class 2: tp=1, fn=1 -> dice 2/3
    assert pc["dice"][2] == pytest.approx(2 / 3)


def test_absent_class_is_not_scored_as_perfect():
    cm = ConfusionMatrix(3)
    cm.update(np.zeros((4, 4), int), np.zeros((4, 4), int))
    s = cm.summary(["background", "grasper", "l_hook"])
    assert s["per_class"]["grasper"]["dice"] is None
    assert s["mean_dice_present_fg"] is None


def test_binary_dice_empty_frames():
    e = np.zeros((5, 5), bool)
    assert binary_dice_iou(e, e) == (None, None)
    m = e.copy()
    m[1, 1] = True
    assert binary_dice_iou(m, e) == (0.0, 0.0)
    assert binary_dice_iou(m, m) == (1.0, 1.0)
