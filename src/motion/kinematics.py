"""Image-space motion and trajectory analysis for tracked keypoints.

Units are image pixels and seconds: displacement [px], speed [px/s],
acceleration [px/s^2], direction [deg]. Without camera calibration and depth
these are NOT physical (mm, mm/s) quantities; the endoscope also moves, so
image motion = instrument motion + camera motion.

Direction convention: image coordinates (x right, y DOWN); 0 deg = moving
right, 90 deg = moving down, +-180 deg = moving left, -90 deg = moving up.

Timestamps come from the video, so dropped frames / irregular sampling are
handled with the true dt. Low-confidence or missing keypoints are skipped;
gaps longer than `max_gap_s` reset the track so no velocity is computed
across them.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MotionState:
    status: str = "missing"        # ok | first_point | missing | low_confidence | gap_reset
    x_smooth: float | None = None
    y_smooth: float | None = None
    dt_s: float | None = None
    displacement_px: float | None = None
    vx_px_s: float | None = None
    vy_px_s: float | None = None
    speed_px_s: float | None = None
    direction_deg: float | None = None
    accel_px_s2: float | None = None
    turn_angle_deg: float | None = None
    # Velocity after removing the estimated camera motion (None if camera motion unknown/unreliable).
    vx_comp_px_s: float | None = None
    vy_comp_px_s: float | None = None
    speed_comp_px_s: float | None = None
    abrupt_change: bool = False
    path_length_px: float = 0.0
    frames_since_valid: int = 0

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


class _Kalman2D:
    """Constant-velocity Kalman filter with per-step dt (adapted from
    files_dataset/tracking.py, which assumed a fixed dt)."""

    def __init__(self, process_var: float, meas_var: float):
        self.q, self.r = process_var, meas_var
        self.x = None
        self.P = None

    def reset(self):
        self.x = None

    def update(self, z: np.ndarray, dt: float) -> np.ndarray:
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.r, self.r, 1e4, 1e4])
            return self.x[:2].copy()
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = self.q * G @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ self.P @ H.T + self.r * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x[:2].copy()


class KeypointMotionTracker:
    """Online motion features for one keypoint."""

    def __init__(self, smoothing: str = "ema", ema_alpha: float = 0.5,
                 kalman_process_var: float = 5e4, kalman_meas_var: float = 25.0,
                 max_gap_s: float = 0.25, min_speed_for_direction: float = 20.0,
                 abrupt_turn_deg: float = 90.0, abrupt_accel_px_s2: float = 8000.0,
                 trail_seconds: float = 2.0):
        if smoothing not in ("none", "ema", "kalman"):
            raise ValueError("smoothing must be none | ema | kalman")
        self.smoothing = smoothing
        self.alpha = float(ema_alpha)
        self.kf = _Kalman2D(kalman_process_var, kalman_meas_var)
        self.max_gap_s = max_gap_s
        self.min_speed_dir = min_speed_for_direction
        self.abrupt_turn = abrupt_turn_deg
        self.abrupt_accel = abrupt_accel_px_s2
        self.trail_seconds = trail_seconds
        self.trail: deque = deque()          # (t, x, y) smoothed, for drawing
        self.path_length = 0.0
        self._gap_seen = False
        self._reset_track()

    def _compose_camera(self, camera) -> None:
        """Accumulate camera motion since the last valid point (None = cannot compensate)."""
        if camera is None or not getattr(self, '_cam_acc', None) is not None:
            self._cam_acc = None if camera is None else getattr(self, '_cam_acc', None)
            return
        if not camera.ok:
            self._cam_acc = None
            return
        self._cam_acc = np.vstack([camera.matrix, [0.0, 0.0, 1.0]]) @ self._cam_acc

    def _reset_track(self):
        self._cam_acc = None
        self.prev_t = None
        self.prev_pos = None
        self.prev_vel = None
        self.kf.reset()
        self.frames_since_valid = 0

    def _smooth(self, z: np.ndarray, dt: float | None) -> np.ndarray:
        if self.smoothing == "none" or self.prev_pos is None:
            if self.smoothing == "kalman":
                return self.kf.update(z, dt or 0.0)
            return z
        if self.smoothing == "ema":
            return self.alpha * z + (1 - self.alpha) * self.prev_pos
        return self.kf.update(z, dt)

    def update(self, t: float, x: float | None, y: float | None, valid: bool,
               low_confidence: bool = False, camera=None) -> MotionState:
        """camera: CameraMotion from the previous frame to this one (optional)."""
        self._compose_camera(camera)
        while self.trail and t - self.trail[0][0] > self.trail_seconds:
            self.trail.popleft()

        if not valid or x is None or y is None:
            self.frames_since_valid += 1
            status = "low_confidence" if low_confidence else "missing"
            if self.prev_t is not None and t - self.prev_t > self.max_gap_s:
                self._reset_track()
                self._gap_seen = True
                self.frames_since_valid = 1
            return MotionState(status=status, path_length_px=self.path_length,
                               frames_since_valid=self.frames_since_valid)

        z = np.array([float(x), float(y)])
        gap = self._gap_seen or (self.prev_t is not None and (t - self.prev_t) > self.max_gap_s)
        self._gap_seen = False
        if gap or (self.prev_t is not None and t <= self.prev_t):
            self._reset_track()
        dt = None if self.prev_t is None else t - self.prev_t
        pos = self._smooth(z, dt)
        st = MotionState(status="gap_reset" if gap else ("first_point" if dt is None else "ok"),
                         x_smooth=float(pos[0]), y_smooth=float(pos[1]), dt_s=dt)

        if dt is not None and dt > 0:
            d = pos - self.prev_pos
            vel = d / dt
            speed = float(np.hypot(*vel))
            st.displacement_px = float(np.hypot(*d))
            st.vx_px_s, st.vy_px_s, st.speed_px_s = float(vel[0]), float(vel[1]), speed
            if speed >= self.min_speed_dir:
                st.direction_deg = math.degrees(math.atan2(vel[1], vel[0]))
            if self.prev_vel is not None:
                st.accel_px_s2 = float(np.hypot(*((vel - self.prev_vel) / dt)))
                ps = float(np.hypot(*self.prev_vel))
                if speed >= self.min_speed_dir and ps >= self.min_speed_dir:
                    cosang = float(np.dot(vel, self.prev_vel) / (speed * ps))
                    st.turn_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                st.abrupt_change = bool(
                    (st.turn_angle_deg is not None and st.turn_angle_deg >= self.abrupt_turn)
                    or st.accel_px_s2 >= self.abrupt_accel)
            self.path_length += st.displacement_px
            self.prev_vel = vel
            if self._cam_acc is not None:
                expected = self._cam_acc[:2, :2] @ self.prev_pos + self._cam_acc[:2, 2]
                vc = (pos - expected) / dt
                st.vx_comp_px_s, st.vy_comp_px_s = float(vc[0]), float(vc[1])
                st.speed_comp_px_s = float(np.hypot(*vc))

        self.prev_t, self.prev_pos = t, pos
        self._cam_acc = np.eye(3) if camera is not None else None
        self.frames_since_valid = 0
        st.path_length_px = self.path_length
        self.trail.append((t, float(pos[0]), float(pos[1])))
        return st

    def trail_points(self) -> list[tuple[int, int]]:
        return [(int(round(x)), int(round(y))) for _, x, y in self.trail]


def compute_kinematics(times, xs, ys, valid=None, **tracker_kwargs) -> list[MotionState]:
    """Offline helper: run the online tracker over arrays (used in tests and
    for re-analysing exported keypoint CSVs)."""
    trk = KeypointMotionTracker(**tracker_kwargs)
    valid = np.ones(len(times), bool) if valid is None else np.asarray(valid, bool)
    return [trk.update(float(t), None if not v else float(x), None if not v else float(y), bool(v))
            for t, x, y, v in zip(times, xs, ys, valid)]
