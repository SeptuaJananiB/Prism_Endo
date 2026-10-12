"""APPROXIMATE image scale (mm per pixel) from the instrument shaft width - no camera calibration needed.

Standard laparoscopic cholecystectomy instruments (grasper, L-hook) have a 5 mm shaft. The shaft's apparent width
near the tip (2 x the largest inscribed radius of the instrument mask within `radius_px` of the tip) gives a local
scale at the instrument's depth: mm_per_px = shaft_mm / width_px.

Limits (why this is an ESTIMATE, roughly +-30-50 %): perspective (tissue may be nearer/farther than the tip), the
segmentation mask edge, tapered insulation near the tip, 10 mm instruments, and occlusion. Pixels stay the primary unit;
values are reported as *_mm_approx and must never be read as calibrated measurements.
"""

from __future__ import annotations

import cv2
import numpy as np


def mm_per_px(instrument_mask: np.ndarray, tip_xy: tuple[float, float], shaft_mm: float = 5.0,
              radius_px: int = 200, min_width_px: float = 8.0, max_width_px: float = 200.0) -> float | None:
    """Local mm-per-pixel at the instrument tip, or None when the shaft width cannot be measured plausibly."""
    if instrument_mask is None or tip_xy is None or tip_xy[0] is None or not np.isfinite(tip_xy[0]):
        return None
    m = (np.asarray(instrument_mask) > 0).astype(np.uint8)
    if m.sum() < 50:
        return None
    n, lab = cv2.connectedComponents(m, connectivity=8)
    h, w = m.shape
    x, y = int(round(min(max(tip_xy[0], 0), w - 1))), int(round(min(max(tip_xy[1], 0), h - 1)))
    # component that contains (or is nearest to) the tip
    win = lab[max(0, y - 15):y + 16, max(0, x - 15):x + 16]
    ids = win[win > 0]
    if ids.size:
        comp = lab == np.bincount(ids).argmax()
    else:
        ys, xs = np.nonzero(lab)
        k = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        comp = lab == lab[ys[k], xs[k]]
    dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.ogrid[:h, :w]
    near = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px ** 2
    vals = dt[near & comp]
    if vals.size == 0:
        return None
    width = 2.0 * float(np.percentile(vals, 98))
    if not (min_width_px <= width <= max_width_px):
        return None
    return shaft_mm / width
