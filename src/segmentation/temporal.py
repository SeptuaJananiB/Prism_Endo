"""Temporal smoothing of segmentation probabilities to reduce mask flicker.

Frame-by-frame segmentation flickers: boundary pixels and small/ambiguous
regions change label from frame to frame even when the scene barely moves.
This keeps an exponential moving average (EMA) of the class probabilities
at model resolution:

    P_t = alpha * softmax_t + (1 - alpha) * warp(P_{t-1})

`warp` optionally applies the estimated camera motion (src/motion/camera.py)
so the history stays aligned with the current frame; when the camera
estimate is unreliable the history is used unwarped. alpha = 1 disables
smoothing. The state is reset after time gaps (clip boundaries).

Trade-off: smoothing adds lag for objects that move independently of the
camera (instruments). The effect on accuracy and flicker was measured on the
held-out clips with scripts/evaluate_temporal.py rather than assumed.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.motion.camera import CameraMotion


def scale_affine(matrix: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Express a full-resolution 2x3 affine in a grid scaled by (sx, sy)."""
    s = np.diag([sx, sy, 1.0])
    m3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return (s @ m3 @ np.linalg.inv(s))[:2]


class TemporalSmoother:
    """EMA over low-resolution probability maps (C, h, w)."""

    def __init__(self, alpha: float = 0.6, motion_compensate: bool = True, reset_gap_s: float = 0.25):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.motion_compensate = motion_compensate
        self.reset_gap_s = reset_gap_s
        self.state = None   # (h, w, C) float32, HWC so cv2.warpAffine can warp all channels at once
        self.t = None

    def reset(self) -> None:
        self.state, self.t = None, None

    def update(self, probs_chw: np.ndarray, t: float, full_hw: tuple[int, int],
               camera: CameraMotion | None = None) -> np.ndarray:
        cur = np.ascontiguousarray(np.transpose(probs_chw, (1, 2, 0)).astype(np.float32))
        gap = self.t is None or (t - self.t) > self.reset_gap_s or t <= self.t
        if self.alpha >= 1.0 or self.state is None or gap or self.state.shape != cur.shape:
            self.state = cur
        else:
            prev = self.state
            if self.motion_compensate and camera is not None and camera.ok:
                h, w = cur.shape[:2]
                m = scale_affine(camera.matrix, w / full_hw[1], h / full_hw[0])
                prev = cv2.warpAffine(prev, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if prev.ndim == 2:  # cv2 drops a singleton channel axis
                    prev = prev[..., None]
            self.state = self.alpha * cur + (1.0 - self.alpha) * prev
        self.t = t
        return np.transpose(self.state, (2, 0, 1))


class TemporalSegmenter:
    """Wraps a SegmentationPredictor; same predict() result type (full-res labels)."""

    def __init__(self, predictor, alpha: float = 0.6, motion_compensate: bool = True,
                 reset_gap_s: float = 0.25):
        self.predictor = predictor
        self.class_names = predictor.class_names
        self.smoother = TemporalSmoother(alpha, motion_compensate, reset_gap_s)

    def reset(self) -> None:
        self.smoother.reset()

    def predict(self, frame_bgr: np.ndarray, t: float = 0.0, camera: CameraMotion | None = None) -> np.ndarray:
        if self.smoother.alpha >= 1.0 or not hasattr(self.predictor, "predict_lowres_probs"):
            return self.predictor.predict(frame_bgr)
        probs = self.predictor.predict_lowres_probs([frame_bgr])[0]
        h, w = frame_bgr.shape[:2]
        smoothed = self.smoother.update(probs, t, (h, w), camera)
        return self.predictor.labels_from_lowres(smoothed[None], (h, w))[0]
