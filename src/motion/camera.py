"""Global camera (endoscope) motion between consecutive frames.

Laparoscopic image motion = instrument motion + camera motion (+ tissue
deformation). This estimates a global 2-D similarity transform (rotation,
uniform scale, translation) from the previous frame to the current one by
tracking background corners with pyramidal Lucas-Kanade optical flow and
fitting with RANSAC. Instrument pixels and the black field-of-view border
are excluded, because instruments move independently of the camera.

It is an approximation: tissue deforms, the camera also moves in depth, and
specular highlights move with the light. When too few consistent features
are found the estimate is reported as unreliable (ok=False) and callers must
not use it. Nothing here is validated against ground-truth camera poses
(none exist for this data); see scripts/evaluate_camera_motion.py for the
consistency checks that were run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class CameraMotion:
    ok: bool
    matrix: np.ndarray          # 2x3, maps previous-frame pixel coords -> current-frame coords
    n_tracked: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    residual_px: float | None = None   # median reprojection error of inliers
    ms: float = 0.0

    @property
    def translation(self) -> tuple[float, float]:
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0]))

    @property
    def scale(self) -> float:
        return float(math.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    def apply(self, x: float, y: float) -> tuple[float, float]:
        m = self.matrix
        return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2])

    def as_dict(self, prefix: str = "camera_") -> dict:
        tx, ty = self.translation
        return {f"{prefix}ok": self.ok, f"{prefix}tx_px": round(tx, 3), f"{prefix}ty_px": round(ty, 3),
                f"{prefix}rot_deg": round(self.rotation_deg, 4), f"{prefix}scale": round(self.scale, 5),
                f"{prefix}inliers": self.n_inliers, f"{prefix}inlier_ratio": round(self.inlier_ratio, 3),
                f"{prefix}residual_px": None if self.residual_px is None else round(self.residual_px, 3)}


class CameraMotionEstimator:
    def __init__(self, scale: float = 0.5, max_corners: int = 400, min_inliers: int = 25,
                 min_inlier_ratio: float = 0.5, ransac_px: float = 2.0, exclude_dilate: int = 15):
        self.scale = scale
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_px = ransac_px
        self.exclude_dilate = exclude_dilate
        self.prev_gray = None
        self.prev_mask = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_mask = None

    def _prepare(self, frame_bgr: np.ndarray, exclude: np.ndarray | None):
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        valid = (gray > 20).astype(np.uint8)  # drop the black field-of-view border
        if exclude is not None:
            ex = cv2.resize(exclude.astype(np.uint8), (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            k = max(3, int(self.exclude_dilate * self.scale) | 1)
            valid[cv2.dilate(ex, np.ones((k, k), np.uint8)) > 0] = 0
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        return gray, valid

    def update(self, frame_bgr: np.ndarray, exclude: np.ndarray | None = None) -> CameraMotion:
        """exclude: HxW bool mask of pixels that must not be tracked (instruments)."""
        t0 = time.perf_counter()
        gray, valid = self._prepare(frame_bgr, exclude)
        prev, prev_valid = self.prev_gray, self.prev_mask
        self.prev_gray, self.prev_mask = gray, valid
        if prev is None or prev.shape != gray.shape:
            return CameraMotion(False, IDENTITY.copy(), ms=1000 * (time.perf_counter() - t0))
        pts = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 7, mask=prev_valid)
        if pts is None or len(pts) < self.min_inliers:
            return CameraMotion(False, IDENTITY.copy(), 0 if pts is None else len(pts),
                                ms=1000 * (time.perf_counter() - t0))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
        good = st.ravel() == 1
        p0, p1 = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
        # Forward-backward check removes unstable tracks (specularities, deforming tissue).
        if len(p0) >= self.min_inliers:
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1.reshape(-1, 1, 2), None,
                                                     winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm(back.reshape(-1, 2) - p0, axis=1)
            keep = (st2.ravel() == 1) & (fb < 1.0)
            p0, p1 = p0[keep], p1[keep]
        n = len(p0)
        if n < self.min_inliers:
            return CameraMotion(False, IDENTITY.copy(), n, ms=1000 * (time.perf_counter() - t0))
        m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_px * self.scale)
        if m is None:
            return CameraMotion(False, IDENTITY.copy(), n, ms=1000 * (time.perf_counter() - t0))
        inl = inl.ravel().astype(bool)
        res = np.linalg.norm((p0[inl] @ m[:, :2].T + m[:, 2]) - p1[inl], axis=1)
        # Back to full-resolution pixel units.
        full = m.copy()
        full[:, 2] /= self.scale
        ratio = float(inl.mean())
        ok = int(inl.sum()) >= self.min_inliers and ratio >= self.min_inlier_ratio
        return CameraMotion(ok, full if ok else IDENTITY.copy(), n, int(inl.sum()), ratio,
                            float(np.median(res)) / self.scale if len(res) else None,
                            1000 * (time.perf_counter() - t0))
