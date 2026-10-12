import numpy as np
import pandas as pd
import pytest

from src.evaluation.keypoints import keypoint_errors, summarize_errors
from src.evaluation.warnings import evaluate_warnings, warning_onsets
from src.risk.learned import attach_labels, features_from_pipeline_csv, train_learned_risk


def test_keypoint_errors_skip_unlabelled_and_low_confidence():
    gt = np.array([[0, 0], [10, 10], [np.nan, np.nan], [5, 5]])
    pred = np.array([[3, 4], [10, 10], [1, 1], [100, 100]])
    conf = np.array([0.9, 0.9, 0.9, 0.1])
    r = keypoint_errors(gt, pred, conf, min_conf=0.6)
    assert r["labelled_frames"] == 3 and r["scored_frames"] == 2
    assert r["confident_on_unlabelled"] == 1
    assert r["detection_recall"] == pytest.approx(2 / 3)
    s = summarize_errors(r["errors"], thresholds_px=(5,))
    assert s["mean_px"] == pytest.approx(2.5)
    assert s["pck@5px"] == pytest.approx(1.0)


def test_warning_lead_time_and_false_alarms():
    t = np.arange(0, 60, 0.5)            # 1 minute at 2 Hz
    active = np.zeros_like(t, bool)
    active[(t >= 18) & (t < 25)] = True  # warns 2 s before event at 20 s
    active[(t >= 40) & (t < 41)] = True  # unrelated -> false alarm
    r = evaluate_warnings(t, active, [(20.0, 24.0)], max_lead_s=5)
    assert r["events_detected"] == 1
    assert r["lead_time_s_mean"] == pytest.approx(2.0)
    assert r["false_alarms"] == 1
    assert len(warning_onsets(t, active)) == 2


def test_missed_event_and_late_warning():
    t = np.arange(0, 30, 1.0)
    active = np.zeros_like(t, bool)
    active[(t >= 12) & (t <= 13)] = True  # warns after event start (10 s)
    r = evaluate_warnings(t, active, [(10.0, 14.0), (25.0, 27.0)], max_lead_s=2)
    assert r["events_detected"] == 1 and r["event_recall"] == 0.5
    assert r["lead_time_s_mean"] == pytest.approx(-2.0)


def _fake_pipeline_csv(path, vid, n, seed):
    """Synthetic per-frame rows in the pipeline's CSV format (TEST FIXTURE ONLY)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 25.0
    dist = np.abs(np.sin(t / 2)) * 0.3
    df = pd.DataFrame({"video_id": vid, "frame": np.arange(n), "source_index": np.arange(n),
                       "timestamp_s": t, "image_diag": 979.0,
                       "tip_kp_status": "ok", "tip_speed_px_s": rng.uniform(0, 300, n),
                       "tip_accel_px_s2": rng.uniform(0, 3000, n), "tip_turn_angle_deg": rng.uniform(0, 90, n),
                       "tip_prox_distance_norm": dist, "tip_prox_inside": False,
                       "tip_prox_approach_speed_px_s": rng.normal(0, 50, n)})
    df.to_csv(path, index=False)
    return t, dist


def test_learned_risk_requires_enough_labels_and_groups(tmp_path):
    feats = []
    labels = []
    for i, vid in enumerate(["v1", "v2", "v3", "v4"]):
        p = tmp_path / f"{vid}.csv"
        t, dist = _fake_pipeline_csv(p, vid, 400, i)
        feats.append(features_from_pipeline_csv(p, window=10))
        # Synthetic labels purely to exercise the code path: "event" when close.
        labels += [{"video_id": vid, "source_index": j, "label": int(d < 0.05)} for j, d in enumerate(dist)]
    data = attach_labels(pd.concat(feats), pd.DataFrame(labels))
    meta = train_learned_risk(data, tmp_path / "model", test_videos=["v4"], window=10)
    assert meta["test_videos"] == ["v4"] and "v4" not in meta["train_videos"]
    assert meta["metrics_test"]["auroc"] > 0.9  # label is a function of a feature here
    with pytest.raises(ValueError, match="2 videos"):
        train_learned_risk(data[data.video_id == "v1"], tmp_path / "m2")
    with pytest.raises(ValueError, match="Not enough"):
        train_learned_risk(data, tmp_path / "m3", test_videos=["v4"], min_positive_train=10 ** 6)
