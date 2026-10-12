"""Automatic tip annotation from instrument masks (used to make PSEUDO labels, never human ground truth).

Improves on mask_tip.estimate_tip in two ways:
  * geodesic far end: distance is measured INSIDE the instrument mask from its field-of-view entry, so a bent L-hook or
    an angled grasper gives the true far end rather than the pixel farthest in a straight line;
  * grasper landmark by definition: "jaw_end" (geodesic far end) or "hinge" (point a jaw-length J back from the far end
    along the instrument, J = alpha * shaft width, i.e. scale-aware).
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.tracking.mask_tip import fov_edge_distance

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def geodesic_distance(comp: np.ndarray, seeds: np.ndarray, step: int = 1) -> np.ndarray:
    """Distance (8-connected, unit steps = chessboard, in full-resolution px) inside bool mask `comp` from bool
    `seeds`; -1 outside or unreachable. step=2 runs the search on a half-resolution mask (~4x faster, distances
    quantised to 2 px) and maps the result back to full resolution."""
    if step <= 1:
        return _geodesic_bfs(comp, seeds)
    h, w = comp.shape
    small = cv2.resize(comp.astype(np.uint8), (w // step, h // step), interpolation=cv2.INTER_NEAREST) > 0
    sseed = cv2.resize(seeds.astype(np.uint8), (w // step, h // step), interpolation=cv2.INTER_NEAREST) > 0
    if not (sseed & small).any():  # seeds lost by subsampling: dilate them first
        sseed = cv2.resize(cv2.dilate(seeds.astype(np.uint8), np.ones((step * 2 + 1,) * 2, np.uint8)),
                           (w // step, h // step), interpolation=cv2.INTER_NEAREST) > 0
    d = _geodesic_bfs(small, sseed)
    full = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(comp & (full >= 0), full * step, -1).astype(np.int32)


def _geodesic_bfs(comp: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    h, w = comp.shape
    dist = np.full((h, w), -1, np.int32)
    q = deque()
    for y, x in zip(*np.nonzero(seeds & comp)):
        dist[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for dy, dx in _NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and comp[yy, xx] and dist[yy, xx] < 0:
                dist[yy, xx] = d
                q.append((yy, xx))
    return dist


def _entry_seeds(comp: np.ndarray, dist_edge: np.ndarray, edge_margin: int) -> tuple[np.ndarray, str]:
    near = comp & (dist_edge <= edge_margin)
    if near.any():
        return near, "fov_entry"
    ys, xs = np.nonzero(comp)  # floating instrument: the principal-axis end nearer the FOV edge is the entry
    pts = np.stack([xs, ys], 1).astype(np.float64)
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    proj = (pts - pts.mean(0)) @ vt[0]
    cand = [int(np.argmin(proj)), int(np.argmax(proj))]
    i = min(cand, key=lambda j: dist_edge[ys[j], xs[j]])
    seed = np.zeros_like(comp)
    seed[ys[i], xs[i]] = True
    return seed, "principal_axis_end"


def tip_from_component(comp: np.ndarray, dist_edge: np.ndarray, landmark: str = "far_end", alpha: float = 1.0,
                       edge_margin: int = 12, band: int = 4, jaw_px: float | None = None,
                       step: int = 1, jaw_rel: float | None = None) -> dict | None:
    """Landmark of ONE instrument component.

    landmark: "far_end" (L-hook tip / grasper jaw end) or "hinge" (grasper jaw hinge): jaw_px pixels back along the
    instrument if given, else alpha * shaft width.
    Returns {x, y, confidence, method, geo_len, shaft_width} or None.
    """
    seeds, how = _entry_seeds(comp, dist_edge, edge_margin)
    dist = geodesic_distance(comp, seeds, step)
    dmax = int(dist.max())
    if dmax < 10:
        return None
    # shaft width ~ pixels per unit of geodesic distance near the entry (area of a band / band length)
    near_band = (dist >= 0) & (dist < min(dmax, 40))
    shaft_w = float(near_band.sum()) / max(1, min(dmax, 40))
    if landmark == "far_end":
        sel = dist >= dmax - band
    else:
        if jaw_rel is not None:  # scale-aware: jaw length = jaw_rel x shaft width (2 x max inscribed radius)
            dtw = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
            jaw_px = jaw_rel * 2.0 * float(np.percentile(dtw[comp], 98))
        target = max(0, dmax - (jaw_px if jaw_px is not None else alpha * shaft_w))
        sel = (dist >= target - band) & (dist <= target + band)
    ys, xs = np.nonzero(sel)
    if len(xs) == 0:
        return None
    reach = dmax / max(shaft_w, 1.0)  # elongation: a stub of instrument at the border is unreliable
    conf = float(np.clip(min(1.0, comp.sum() / 3000) * min(1.0, reach / 3.0) * (1.0 if how == "fov_entry" else 0.6),
                         0, 1))
    sy, sx = np.nonzero(seeds)
    return {"x": float(xs.mean()), "y": float(ys.mean()), "confidence": conf, "method": f"{how}+geodesic_{landmark}",
            "entry": (float(sx.mean()), float(sy.mean())), "area": int(comp.sum()),
            "geo_len": dmax, "shaft_width": round(shaft_w, 1)}


def components(mask: np.ndarray, min_area: int = 300, keep: int = 2) -> list[np.ndarray]:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    return [lab == i for i in order[:keep] if st[i, cv2.CC_STAT_AREA] >= min_area]


def auto_tips(label: np.ndarray, fov: np.ndarray, grasper_landmark: str = "hinge", alpha: float = 0.5,
              jaw_px: float | None = None, lhook_id: int = 32, grasper_id: int = 31) -> dict:
    """{lhook_tip, grasper_tip, grasper_tip_2} -> dict or None, from a CholecSeg8k-style label image."""
    de = fov_edge_distance(fov)
    out = {"lhook_tip": None, "grasper_tip": None, "grasper_tip_2": None}
    lh = components(label == lhook_id, keep=1)
    if lh:
        out["lhook_tip"] = tip_from_component(lh[0], de, "far_end")
    found = []
    for c in components(label == grasper_id):
        t = tip_from_component(c, de, grasper_landmark, alpha, jaw_px=jaw_px)
        if t:
            ys, xs = np.nonzero(c & (de <= 12))
            t["entry_x"] = float(xs.mean()) if len(xs) else t["x"]
            found.append(t)
    found.sort(key=lambda t: t["entry_x"])  # grasper_tip = the one entering further left (annotation guide)
    for name, t in zip(("grasper_tip", "grasper_tip_2"), found):
        out[name] = t
    return out
