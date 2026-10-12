"""Segmentation metrics from an accumulated confusion matrix.

Dice and IoU are computed per class over the whole evaluation set
(dataset-level, the usual convention for CholecSeg8k/EndoVis reporting).
A class with no ground-truth *and* no predicted pixels in the evaluated set
gets ``None`` (not a perfect score), so absent classes are never reported
as successes. Pixels labelled IGNORE_INDEX are excluded.
"""

from __future__ import annotations

import numpy as np

from src.data.segmentation_data import IGNORE_INDEX


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.n = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)
        # Per-frame bookkeeping for "is there any instrument?" style analysis.
        self.frames = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        valid = target != IGNORE_INDEX
        idx = target[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)
        self.frames += 1

    def per_class(self) -> dict[str, list]:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        dice, iou = [], []
        for c in range(self.n):
            denom_d = 2 * tp[c] + fp[c] + fn[c]
            denom_i = tp[c] + fp[c] + fn[c]
            dice.append(None if denom_d == 0 else float(2 * tp[c] / denom_d))
            iou.append(None if denom_i == 0 else float(tp[c] / denom_i))
        gt_pixels = self.mat.sum(1).tolist()
        return {"dice": dice, "iou": iou, "gt_pixels": gt_pixels,
                "pred_pixels": self.mat.sum(0).tolist()}

    def summary(self, names: list[str], exclude_background: bool = True) -> dict:
        pc = self.per_class()
        rows = {}
        for i, name in enumerate(names):
            rows[name] = {"dice": pc["dice"][i], "iou": pc["iou"][i],
                          "gt_pixels": int(pc["gt_pixels"][i]),
                          "present_in_ground_truth": pc["gt_pixels"][i] > 0}
        start = 1 if exclude_background else 0
        # Mean only over classes that actually occur in the ground truth.
        valid = [i for i in range(start, self.n) if pc["gt_pixels"][i] > 0 and pc["dice"][i] is not None]
        mean_dice = float(np.mean([pc["dice"][i] for i in valid])) if valid else None
        mean_iou = float(np.mean([pc["iou"][i] for i in valid])) if valid else None
        total = self.mat.sum()
        acc = float(np.diag(self.mat).sum() / total) if total else None
        return {"per_class": rows, "mean_dice_present_fg": mean_dice,
                "mean_iou_present_fg": mean_iou, "pixel_accuracy": acc,
                "frames": self.frames}


def binary_dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float | None, float | None]:
    """Dice/IoU for one binary mask pair; (None, None) if both are empty."""
    pred = np.asarray(pred, bool)
    target = np.asarray(target, bool)
    inter = np.logical_and(pred, target).sum()
    s = pred.sum() + target.sum()
    union = np.logical_or(pred, target).sum()
    if s == 0:
        return None, None
    return float(2 * inter / s), float(inter / union)
