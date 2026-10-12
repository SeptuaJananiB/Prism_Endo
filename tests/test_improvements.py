"""Tests for temporal smoothing, camera-motion estimation/compensation and
multi-instance tip tracking (synthetic data with known answers)."""

import cv2
import numpy as np
import pytest

from src.motion.camera import IDENTITY, CameraMotion, CameraMotionEstimator
from src.motion.kinematics import KeypointMotionTracker
from src.segmentation.temporal import TemporalSmoother, scale_affine
from src.tracking.keypoints import KeypointSpec
from src.tracking.sources import MaskTipKeypointSource


def textured(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2)
    return cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


def test_camera_estimator_recovers_known_translation():
    big = textured(300, 400)
    est = CameraMotionEstimator()
    est.update(big[20:260, 20:340])
    cm = est.update(big[20:260, 26:346])  # view moved 6 px right -> content shifts 6 px left
    assert cm.ok
    tx, ty = cm.translation
    assert tx == pytest.approx(-6.0, abs=0.5) and ty == pytest.approx(0.0, abs=0.5)
    assert cm.residual_px < 1.0


def test_camera_estimator_flags_unreliable_on_blank_frames():
    est = CameraMotionEstimator()
    blank = np.full((240, 320, 3), 128, np.uint8)
    est.update(blank)
    cm = est.update(blank)
    assert not cm.ok and np.allclose(cm.matrix, IDENTITY)


def test_compensated_speed_removes_camera_motion():
    # Tip static relative to the scene; the whole image shifts +10 px/frame in x.
    cam = CameraMotion(True, np.array([[1.0, 0, 10.0], [0, 1.0, 0]]))
    trk = KeypointMotionTracker(smoothing="none")
    trk.update(0.0, 100, 50, True, camera=cam)
    st = trk.update(0.04, 110, 50, True, camera=cam)
    assert st.speed_px_s == pytest.approx(250.0)       # raw image speed
    assert st.speed_comp_px_s == pytest.approx(0.0, abs=1e-6)
    unreliable = CameraMotion(False, IDENTITY.copy())
    st = trk.update(0.08, 120, 50, True, camera=unreliable)
    assert st.speed_comp_px_s is None                   # never compensate with a bad estimate
    st = KeypointMotionTracker(smoothing="none")
    st.update(0.0, 0, 0, True)
    assert st.update(0.04, 4, 0, True).speed_comp_px_s is None  # feature off -> no column value


def test_smoother_alpha1_is_identity_and_ema_reduces_flicker():
    rng = np.random.default_rng(0)
    base = np.zeros((2, 8, 8), np.float32)
    base[1] = 0.55
    base[0] = 0.45
    noisy = [np.clip(base + rng.normal(0, 0.1, base.shape), 0, 1) for _ in range(30)]
    off, on = TemporalSmoother(1.0), TemporalSmoother(0.4)
    flips_off = flips_on = 0
    prev_off = prev_on = None
    for i, p in enumerate(noisy):
        a = off.update(p, i * 0.04, (8, 8)).argmax(0)
        b = on.update(p, i * 0.04, (8, 8)).argmax(0)
        assert np.allclose(off.state.transpose(2, 0, 1), p)
        if prev_off is not None:
            flips_off += (a != prev_off).sum()
            flips_on += (b != prev_on).sum()
        prev_off, prev_on = a, b
    assert flips_on < flips_off / 2


def test_smoother_resets_after_gap_and_scales_affine():
    sm = TemporalSmoother(0.5)
    sm.update(np.ones((1, 4, 4), np.float32), 0.0, (4, 4))
    out = sm.update(np.zeros((1, 4, 4), np.float32), 1.0, (4, 4))  # 1 s gap -> no blending
    assert out.max() == 0.0
    m = scale_affine(np.array([[1.0, 0, 10.0], [0, 1.0, 20.0]]), 0.5, 0.25)
    assert m[0, 2] == pytest.approx(5.0) and m[1, 2] == pytest.approx(5.0)


def two_graspers(y1, y2, h=200, w=300):
    lab = np.zeros((h, w), np.uint8)
    lab[y1:y1 + 10, 0:150] = 1   # enters from the left
    lab[y2:y2 + 10, 150:300] = 1  # enters from the right
    return lab


def test_multi_instance_slots_keep_identity():
    src = MaskTipKeypointSource([KeypointSpec("grasper_tip", "grasper", max_instances=2)],
                                ["background", "grasper"], min_area=100)
    img = np.full((200, 300, 3), 120, np.uint8)
    first = {k.name: k for k in src(img, 0, two_graspers(40, 150))}
    assert set(first) == {"grasper_tip", "grasper_tip_2"}
    assert all(k.status == "ok" for k in first.values())
    left = min(first.values(), key=lambda k: k.x).name
    # Both move a little; the left-entering instrument must stay in the same slot.
    for step in range(1, 5):
        kps = {k.name: k for k in src(img, step, two_graspers(40 + 3 * step, 150 - 3 * step))}
        assert kps[left].x < 160
    # One instrument leaves: one slot becomes missing, nothing is invented.
    lab = np.zeros((200, 300), np.uint8)
    lab[60:70, 0:150] = 1
    kps = {k.name: k for k in src(img, 9, lab)}
    assert kps[left].status == "ok"
    assert [k.status for n, k in kps.items() if n != left] == ["missing"]
