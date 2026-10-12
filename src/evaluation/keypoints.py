"""Keypoint accuracy against human labels (DLC CollectedData tables).

Only frames where the ground-truth point is labelled are scored; frames
where it is unlabelled (NaN) are counted separately, so a prediction for an
invisible point is reported as a possible false positive rather than an
error measurement.

Metrics per keypoint:
  * mean / median / 90th-percentile pixel error
  * PCK@t: fraction of labelled frames with error <= t pixels
  * PCK@f%diag: same with t = f * image diagonal
  * detection: how often a confident prediction exists when the point is
    labelled (recall) / unlabelled (false positives)
"""

from __future__ import annotations

import math

import numpy as np


def keypoint_errors(gt_xy: np.ndarray, pred_xy: np.ndarray, pred_conf: np.ndarray,
                    min_conf: float = 0.6) -> dict:
    """gt_xy, pred_xy: (N, 2) with NaN for missing; pred_conf: (N,)."""
    gt_xy, pred_xy = np.asarray(gt_xy, float), np.asarray(pred_xy, float)
    pred_conf = np.asarray(pred_conf, float)
    labelled = np.isfinite(gt_xy).all(1)
    confident = np.isfinite(pred_xy).all(1) & (pred_conf >= min_conf)
    both = labelled & confident
    err = np.hypot(*(pred_xy[both] - gt_xy[both]).T) if both.any() else np.array([])
    return {
        "labelled_frames": int(labelled.sum()),
        "unlabelled_frames": int((~labelled).sum()),
        "scored_frames": int(both.sum()),
        "detection_recall": float(both.sum() / labelled.sum()) if labelled.any() else None,
        "confident_on_unlabelled": int((confident & ~labelled).sum()),
        "errors": err,
    }


def summarize_errors(err: np.ndarray, thresholds_px=(5, 10, 20), image_diag: float | None = None,
                     diag_fracs=(0.01, 0.02, 0.05)) -> dict:
    if len(err) == 0:
        return {"mean_px": None, "median_px": None, "p90_px": None}
    out = {"mean_px": float(err.mean()), "median_px": float(np.median(err)),
           "p90_px": float(np.percentile(err, 90))}
    for t in thresholds_px:
        out[f"pck@{t}px"] = float((err <= t).mean())
    if image_diag:
        for f in diag_fracs:
            out[f"pck@{f * 100:g}%diag"] = float((err <= f * image_diag).mean())
    return out


def evaluate_keypoints(gt: dict, pred: dict, frames: list[int], bodyparts: list[str],
                       min_conf: float = 0.6, image_size: tuple[int, int] | None = None) -> dict:
    """gt/pred: objects with .get(frame, bodypart) -> (x, y, conf) | None
    (e.g. src.tracking.dlc_io.DLCPredictions)."""
    diag = math.hypot(*image_size) if image_size else None
    results = {}
    for bp in bodyparts:
        g, p, c = [], [], []
        for f in frames:
            gv, pv = gt.get(f, bp), pred.get(f, bp)
            g.append(gv[:2] if gv else (np.nan, np.nan))
            p.append(pv[:2] if pv else (np.nan, np.nan))
            c.append(pv[2] if pv else 0.0)
        r = keypoint_errors(np.array(g), np.array(p), np.array(c), min_conf)
        err = r.pop("errors")
        r.update(summarize_errors(err, image_diag=diag))
        results[bp] = r
    return results
