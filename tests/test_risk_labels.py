import numpy as np

from src.risk.labels import (annotation_path, append_annotation, frame_kappa, merge_intervals, read_annotation_file,
                             read_label_files, training_gate, union_length, validate_labels)


def test_append_and_read_annotation_roundtrip(tmp_path):
    p = annotation_path(tmp_path, "Dr X")
    assert p.name == "risk_events_Dr_X.csv"
    append_annotation(p, {"video_id": "v1", "start_s": 1.0, "end_s": 2.5, "label": 1, "annotator": "Dr X"})
    df = read_annotation_file(p)
    assert len(df) == 1 and df.loc[0, "video_id"] == "v1" and df.loc[0, "created_at"]


def test_interval_helpers():
    assert merge_intervals([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert abs(union_length([(0, 1), (0.5, 2)]) - 2.0) < 1e-9


def test_kappa_perfect_agreement():
    a = np.array([0, 1, 1, 0, 1])
    assert abs(frame_kappa(a, a.copy())["kappa"] - 1.0) < 1e-9


def test_validator_rejects_end_before_start(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("video_id,start_s,end_s,label,annotator\nv1,5.0,2.0,1,a\n")
    rep = validate_labels(read_label_files([p]))
    assert not rep.ok


def test_validator_rejects_missing_columns(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("video_id,label\nv1,1\n")
    assert not validate_labels(read_label_files([p])).ok


def test_training_gate_refuses_without_labels(tmp_path):
    res = training_gate([str(tmp_path / "none.csv")], [])
    assert not res.ok
