import math

import numpy as np
import pytest

from src.motion.kinematics import KeypointMotionTracker, compute_kinematics


def test_constant_velocity_speed_and_direction():
    t = np.arange(10) / 25.0
    xs = 100 + 1250.0 * t             # 1250 px/s to the right (50 px per frame at 25 fps)
    ys = np.full_like(t, 200.0)
    st = compute_kinematics(t, xs, ys, smoothing="none")
    assert st[0].status == "first_point" and st[0].speed_px_s is None
    for s in st[1:]:
        assert s.speed_px_s == pytest.approx(1250.0, rel=1e-6)
        assert s.direction_deg == pytest.approx(0.0, abs=1e-6)
        assert s.displacement_px == pytest.approx(50.0, rel=1e-6)
    assert st[-1].path_length_px == pytest.approx(450.0, rel=1e-6)
    # Constant velocity: no acceleration, no turning.
    assert st[-1].accel_px_s2 == pytest.approx(0.0, abs=1e-6)
    assert st[-1].turn_angle_deg == pytest.approx(0.0, abs=1e-4)


def test_direction_convention_image_y_down():
    st = compute_kinematics([0.0, 0.1], [0, 0], [0, 10], smoothing="none")
    assert st[1].direction_deg == pytest.approx(90.0)  # moving down the image


def test_irregular_timestamps_use_true_dt():
    # Frame 2 arrives 0.2 s later (dropped frames): speed must use 0.2 s.
    st = compute_kinematics([0.0, 0.04, 0.24], [0, 4, 24], [0, 0, 0], smoothing="none", max_gap_s=0.5)
    assert st[1].speed_px_s == pytest.approx(100.0)
    assert st[2].speed_px_s == pytest.approx(100.0)
    assert st[2].dt_s == pytest.approx(0.2)


def test_missing_and_low_confidence_points_are_skipped():
    trk = KeypointMotionTracker(smoothing="none", max_gap_s=1.0)
    trk.update(0.0, 0, 0, True)
    s = trk.update(0.04, None, None, False)
    assert s.status == "missing" and s.speed_px_s is None
    s = trk.update(0.08, 999, 999, False, low_confidence=True)
    assert s.status == "low_confidence"
    s = trk.update(0.12, 12, 0, True)  # velocity spans the gap (0.12 s < max_gap)
    assert s.speed_px_s == pytest.approx(100.0)


def test_long_gap_resets_track():
    trk = KeypointMotionTracker(smoothing="none", max_gap_s=0.25)
    trk.update(0.0, 0, 0, True)
    trk.update(0.5, None, None, False)
    s = trk.update(1.0, 500, 0, True)
    assert s.status == "gap_reset"
    assert s.speed_px_s is None  # no velocity across a 1 s gap


def test_abrupt_reversal_detected():
    t = np.arange(6) / 25.0
    xs = [0, 10, 20, 30, 20, 10]  # reverses direction at frame 4
    st = compute_kinematics(t, xs, [0] * 6, smoothing="none")
    assert st[4].turn_angle_deg == pytest.approx(180.0)
    assert st[4].abrupt_change
    assert not st[3].abrupt_change


def test_slow_motion_has_no_direction():
    st = compute_kinematics([0, 0.04], [0, 0.1], [0, 0], smoothing="none", min_speed_for_direction=20)
    assert st[1].speed_px_s == pytest.approx(2.5)
    assert st[1].direction_deg is None


@pytest.mark.parametrize("smoothing", ["ema", "kalman"])
def test_smoothing_reduces_jitter(smoothing):
    rng = np.random.default_rng(0)
    t = np.arange(100) / 25.0
    xs = 200 + rng.normal(0, 3, 100)  # stationary point with noise
    raw = compute_kinematics(t, xs, np.full(100, 100.0), smoothing="none")
    smo = compute_kinematics(t, xs, np.full(100, 100.0), smoothing=smoothing)
    raw_speed = np.mean([s.speed_px_s for s in raw[5:]])
    smo_speed = np.mean([s.speed_px_s for s in smo[5:]])
    assert smo_speed < raw_speed
    assert all(math.isfinite(s.x_smooth) for s in smo)
