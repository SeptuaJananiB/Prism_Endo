import math

import pytest

from src.motion.kinematics import MotionState
from src.proximity.proximity import ProximityResult
from src.risk.rule_based import RuleBasedRiskScorer, combine
from src.tracking.keypoints import Keypoint
from src.utils.config import load_config, resolve_path

CFG = load_config(resolve_path("configs/pipeline.yaml"))["risk"]
DIAG = math.hypot(854, 480)


def kp(valid=True):
    k = Keypoint("tip", 100.0, 100.0, 0.9, "mask_heuristic", "ok" if valid else "low_confidence")
    return k


def motion(speed=0.0, accel=0.0, turn=None):
    return MotionState(status="ok", x_smooth=100, y_smooth=100, vx_px_s=speed, vy_px_s=0.0,
                       speed_px_s=speed, accel_px_s2=accel, turn_angle_deg=turn)


def prox(dist_px, inside=False, approach=0.0):
    return ProximityResult("liver", "ok", inside, dist_px, -dist_px if inside else dist_px, dist_px / DIAG,
                           0, 0, approach)


def test_score_increases_as_tip_gets_closer():
    scores = []
    for d in (300, 100, 40, 10, 0):
        s = RuleBasedRiskScorer(CFG, "tip")
        scores.append(s.update(0.0, kp(), motion(), prox(d), True, DIAG).raw_score)
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.0)
    assert scores[-1] == pytest.approx(CFG["weights"]["proximity"])


def test_inside_target_gives_full_proximity_term():
    s = RuleBasedRiskScorer(CFG, "tip")
    out = s.update(0.0, kp(), motion(), prox(5, inside=True), True, DIAG)
    assert out.contributions["proximity"] == pytest.approx(CFG["weights"]["proximity"])


def test_motion_only_matters_near_tissue():
    far = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(speed=2000), prox(500), True, DIAG)
    near = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(speed=2000), prox(20), True, DIAG)
    assert far.contributions["speed"] == pytest.approx(0.0)
    assert near.contributions["speed"] > 0.0


def test_approach_only_counts_towards_tissue():
    towards = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(), prox(20, approach=300), True, DIAG)
    away = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(), prox(20, approach=-300), True, DIAG)
    assert towards.contributions["approach"] > 0
    assert away.contributions["approach"] == 0


def test_score_bounded_0_1():
    out = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(speed=1e6, accel=1e9, turn=180),
                                                 prox(0, inside=True, approach=1e6), True, DIAG)
    assert 0.0 <= out.raw_score <= 1.0
    assert out.raw_score == pytest.approx(1.0)


def test_missing_keypoint_gives_no_score():
    out = RuleBasedRiskScorer(CFG, "tip").update(0.0, None, None, None, True, DIAG)
    assert out.score is None and out.status == "no_keypoint" and out.category == "Unavailable"
    out = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(valid=False), None, None, True, DIAG)
    assert out.status == "low_confidence" and out.score is None


def test_motion_only_mode_and_disable_mode():
    out = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(speed=500), None, False, DIAG)
    assert out.mode == "motion_only" and "tissue_model" in out.missing
    assert "proximity" not in out.contributions
    assert "motion-only" in out.explanation
    cfg = dict(CFG, when_proximity_unavailable="disable")
    out = RuleBasedRiskScorer(cfg, "tip").update(0.0, kp(), motion(speed=500), None, False, DIAG)
    assert out.mode == "disabled" and out.score is None


def test_target_not_visible_is_flagged():
    absent = ProximityResult("liver", "absent")
    out = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(), absent, True, DIAG)
    assert "target_tissue_not_visible" in out.missing
    assert out.raw_score == 0.0


def test_dwell_time_prevents_single_frame_escalation():
    s = RuleBasedRiskScorer(dict(CFG, smoothing_alpha=1.0, rapid_rise_per_s=1e9, dwell_s=0.2), "tip")
    t = 0.0
    for _ in range(5):
        s.update(t, kp(), motion(), prox(500), True, DIAG)
        t += 0.04
    spike = s.update(t, kp(), motion(), prox(0, inside=True), True, DIAG)
    assert spike.raw_score >= 0.5
    assert spike.category == "Minimal"  # one frame is not enough
    for _ in range(6):
        t += 0.04
        out = s.update(t, kp(), motion(), prox(0, inside=True), True, DIAG)
    assert out.category in ("Medium", "High")


def test_rapid_rise_escalates_immediately():
    s = RuleBasedRiskScorer(dict(CFG, smoothing_alpha=1.0, rapid_rise_per_s=1.0, dwell_s=5.0), "tip")
    s.update(0.0, kp(), motion(), prox(500), True, DIAG)
    out = s.update(0.04, kp(), motion(), prox(0, inside=True), True, DIAG)
    assert out.category != "Minimal"


def test_explanation_lists_contributing_factors():
    out = RuleBasedRiskScorer(CFG, "tip").update(0.0, kp(), motion(speed=400), prox(10, approach=200), True, DIAG)
    assert "liver" in out.explanation and "px" in out.explanation


def test_combine_takes_max_and_handles_empty():
    a = RuleBasedRiskScorer(CFG, "a").update(0.0, kp(), motion(), prox(200), True, DIAG)
    b = RuleBasedRiskScorer(CFG, "b").update(0.0, kp(), motion(), prox(5), True, DIAG)
    assert combine([a, b]).keypoint == "b"
    none = RuleBasedRiskScorer(CFG, "c").update(0.0, None, None, None, True, DIAG)
    assert combine([none]) is None
