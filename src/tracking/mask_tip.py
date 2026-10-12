"""Heuristic instrument-tip estimate from a segmentation mask (fallback only).

Idea: laparoscopic instruments enter the field of view (FOV) from its edge,
so the tip is the part of the instrument mask farthest from where the mask
touches the FOV boundary. When the mask does not touch the boundary, the tip
is taken as the end of the mask's principal axis lying deeper inside the
FOV, with reduced quality score.

This is NOT a substitute for trained keypoint tracking (DeepLabCut). It can
be wrong for curved instruments, crossing instruments, or masks that merge
two tools. Its output `confidence` is a heuristic quality score in [0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np


def field_of_view_mask(bgr: np.ndarray, thresh: int = 20) -> np.ndarray:
    """Binary mask of the illuminated endoscope field (excludes black borders)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fov = (gray > thresh).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fov, 8)
    if n <= 1:
        return np.ones_like(fov, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == largest


def fov_edge_distance(fov: np.ndarray) -> np.ndarray:
    """Distance of every pixel to the field-of-view boundary (image border counts as boundary)."""
    fov_u8 = np.pad(fov.astype(np.uint8), 1, constant_values=0)
    return cv2.distanceTransform(fov_u8, cv2.DIST_L2, 5)[1:-1, 1:-1]


def estimate_tip(mask: np.ndarray, fov: np.ndarray | None = None, min_area: int = 300,
                 edge_margin: int = 12, ref_area: int = 4000,
                 dist_edge: np.ndarray | None = None) -> dict | None:
    """Return {"x","y","confidence","entry","method"} or None if no usable mask.

    mask : HxW bool/uint8 mask of ONE instrument class.
    fov  : HxW bool field-of-view mask (None = whole image).
    dist_edge : optional precomputed fov_edge_distance(fov) (the FOV is static, so callers cache it).
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    total = int(mask.sum())
    if total < min_area:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    comp = lab == largest
    ys, xs = np.nonzero(comp)

    h, w = mask.shape
    if dist_edge is None:
        dist_edge = fov_edge_distance(fov if fov is not None else np.ones((h, w), bool))

    near_edge = dist_edge[ys, xs] <= edge_margin
    dominance = area / total
    size_q = min(1.0, area / ref_area)
    if near_edge.any():
        entry = np.array([xs[near_edge].mean(), ys[near_edge].mean()])
        d = np.hypot(xs - entry[0], ys - entry[1])
        method, method_q = "farthest_from_fov_entry", 1.0
    else:
        pts = np.stack([xs, ys], 1).astype(np.float64)
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        proj = (pts - mean) @ vt[0]
        end_a, end_b = pts[np.argmin(proj)], pts[np.argmax(proj)]
        # The end deeper inside the FOV is taken as the tip.
        da = dist_edge[int(end_a[1]), int(end_a[0])]
        db = dist_edge[int(end_b[1]), int(end_b[0])]
        entry = end_a if da < db else end_b
        d = np.hypot(xs - entry[0], ys - entry[1])
        method, method_q = "principal_axis_end", 0.6

    # Average the few farthest pixels to reduce single-pixel noise.
    k = max(1, min(25, len(d) // 200))
    far = np.argpartition(d, -k)[-k:]
    tip = (float(xs[far].mean()), float(ys[far].mean()))
    conf = float(np.clip(size_q * dominance * method_q, 0.0, 1.0))
    return {"x": tip[0], "y": tip[1], "confidence": conf,
            "entry": (float(entry[0]), float(entry[1])), "method": method, "area": area}
