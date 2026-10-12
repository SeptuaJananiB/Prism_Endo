"""Instrument-tip to tissue-boundary proximity in IMAGE SPACE (Stage E).

For each configured tissue class this computes, for one keypoint:
  * whether the tip pixel lies inside the class mask,
  * the minimum Euclidean distance (pixels) from the tip to the class
    boundary, and the boundary point that achieves it,
  * a signed distance (negative = inside the region),
  * the distance normalised by the image diagonal (resolution independent),
  * the approach speed: the component of the tip velocity pointing towards
    that boundary point (px/s, positive = approaching).

These are 2-D image distances. They are NOT millimetres: without depth,
stereo or calibration, an image distance of 10 px can correspond to very
different physical gaps. Also, the instrument occludes tissue, so "distance
to the nearest VISIBLE pixel of the class" is what is measured.

Edge cases handled: no tissue model (status "unavailable"), class absent or
tiny in this frame ("absent"), invalid keypoint ("no_keypoint"), keypoint
outside the image, and masks touching the image border (border pixels are
not treated as tissue boundary).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class ProximityResult:
    target: str
    status: str                        # ok | absent | unavailable | no_keypoint
    inside: bool | None = None
    distance_px: float | None = None
    signed_distance_px: float | None = None
    distance_norm: float | None = None  # distance / image diagonal
    closest_x: float | None = None
    closest_y: float | None = None
    approach_speed_px_s: float | None = None

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def boundary_points(mask: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """(N, 2) array of x, y boundary pixels of a binary mask, excluding pixels
    on the image border and pixels inside `exclude` (e.g. the dilated black
    field-of-view border): those are frame edges, not tissue boundaries."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), np.float64)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours], 0)
    h, w = mask.shape
    keep = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
    if exclude is not None:
        keep &= ~exclude[pts[:, 1], pts[:, 0]]
    return pts[keep].astype(np.float64) if keep.any() else pts.astype(np.float64)


def _query(mask: np.ndarray, pts: np.ndarray, x: float, y: float, target: str,
           velocity: tuple[float, float] | None) -> ProximityResult:
    if not (math.isfinite(x) and math.isfinite(y)):
        return ProximityResult(target, "no_keypoint")
    if len(pts) == 0:
        return ProximityResult(target, "absent")
    h, w = mask.shape
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    i = int(np.argmin(d))
    dist = float(d[i])
    xi, yi = int(round(x)), int(round(y))
    inside = bool(0 <= xi < w and 0 <= yi < h and mask[yi, xi])
    diag = math.hypot(w, h)
    res = ProximityResult(target, "ok", inside, dist, -dist if inside else dist, dist / diag,
                          float(pts[i, 0]), float(pts[i, 1]))
    if velocity is not None and all(v is not None for v in velocity) and dist > 1e-6 and not inside:
        ux, uy = (pts[i, 0] - x) / dist, (pts[i, 1] - y) / dist
        res.approach_speed_px_s = float(velocity[0] * ux + velocity[1] * uy)
    return res


def proximity_to_mask(mask: np.ndarray, x: float, y: float, target: str = "target",
                      velocity: tuple[float, float] | None = None, min_area: int = 50,
                      exclude: np.ndarray | None = None) -> ProximityResult:
    mask = np.asarray(mask, bool)
    if mask.sum() < min_area:
        return ProximityResult(target, "absent")
    if not (math.isfinite(x) and math.isfinite(y)):
        return ProximityResult(target, "no_keypoint")
    return _query(mask, boundary_points(mask, exclude), x, y, target, velocity)


class ProximityEstimator:
    """Per frame: boundaries of every target are extracted once (prepare) and
    then queried for each keypoint. Boundary pixels touching the black
    field-of-view border (tissue class index `border_class`) are ignored."""

    def __init__(self, target_classes: list[str], min_area: int = 50, border_class: int | None = 0,
                 border_margin: int = 3):
        self.targets = list(target_classes)
        self.min_area = min_area
        self.border_class = border_class
        self.border_margin = border_margin
        self._labels_ref = None  # the label map the cache was built from (held, so identity is safe)
        self._prepared: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def unavailable(self) -> list[ProximityResult]:
        return [ProximityResult(t, "unavailable") for t in self.targets]

    def prepare(self, tissue_labels: np.ndarray, class_names: list[str]) -> None:
        exclude = None
        if self.border_class is not None:
            border = (tissue_labels == self.border_class).astype(np.uint8)
            if border.any():
                k = 2 * self.border_margin + 1
                exclude = cv2.dilate(border, np.ones((k, k), np.uint8)).astype(bool)
        self._prepared = {}
        for t in self.targets:
            m = tissue_labels == class_names.index(t)
            self._prepared[t] = (m, boundary_points(m, exclude)) if m.sum() >= self.min_area else None
        self._labels_ref = tissue_labels

    def __call__(self, tissue_labels: np.ndarray | None, class_names: list[str],
                 x: float | None, y: float | None,
                 velocity: tuple[float, float] | None = None) -> list[ProximityResult]:
        if tissue_labels is None:
            return self.unavailable()
        if x is None or y is None:
            return [ProximityResult(t, "no_keypoint") for t in self.targets]
        if self._labels_ref is not tissue_labels:  # new frame / label map: extract boundaries once
            self.prepare(tissue_labels, class_names)
        out = []
        for t in self.targets:
            prep = self._prepared[t]
            out.append(ProximityResult(t, "absent") if prep is None else _query(prep[0], prep[1], x, y, t, velocity))
        return out


def nearest(results: list[ProximityResult]) -> ProximityResult | None:
    ok = [r for r in results if r.status == "ok"]
    return min(ok, key=lambda r: r.signed_distance_px) if ok else None
