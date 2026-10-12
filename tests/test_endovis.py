import json

import numpy as np

from src.external.endovis import FrameLabels, bodyparts, load_labels, to_dlc_table, validate


def write(tmp_path, entries):
    p = tmp_path / "train9_labels.json"
    p.write_text(json.dumps(entries))
    return p


def entry(n, anns):
    return {"class": "image", "filename": f"/x/Training/Dataset1/Raw/img_{n:06d}_raw.png", "annotations": anns}


def a(tool, joint, x, y):
    return {"class": joint, "id": tool, "x": x, "y": y}


def test_parse_flags_duplicates_and_unknown(tmp_path):
    p = write(tmp_path, [entry(1, []),
                         entry(2, [a("tool1", "HeadPoint", 10, 20), a("tool1", "HeadPoint", 11, 21),
                                   a("tool9", "Foo", 1, 1)])])
    frames, st = load_labels(p, "train")
    assert st == {"entries": 2, "labelled_frames": 1}
    f = frames[1]
    assert f.frame_number == 2 and f.points[("tool1", "HeadPoint")] == (10.0, 20.0)  # first kept, not repaired
    assert any("duplicate" in i for i in f.issues) and any("unknown tool" in i for i in f.issues)


def test_validate_range_and_missing(tmp_path, monkeypatch):
    import src.external.endovis as ev
    monkeypatch.setattr(ev, "video_info", lambda p: {"frames_decoded": 5})
    ok = FrameLabels("train", 1, 3, {("tool1", j): (10.0, 10.0) for j in ev.JOINTS})
    out = FrameLabels("train", 1, 9, {("tool1", "HeadPoint"): (10.0, 10.0)})
    bad = FrameLabels("train", 1, 4, {("tool2", "HeadPoint"): (900.0, 10.0)})
    rep = validate([ok, out, bad], {1: "v.avi"})
    assert rep["summary"]["matched"] == 2 and rep["summary"]["unmatched_frame_out_of_range"] == 1
    issues = rep["flagged"][0]["issues"]
    assert any("outside image" in i for i in issues) and any("missing joints" in i for i in issues)


def test_dlc_table_keeps_source_names_and_nan():
    f = FrameLabels("train", 1, 60, {("tool2", "LeftClasperPoint"): (345.6, 221.3)})
    df = to_dlc_table([f], "du2018", "endovis_seq1", 1107)
    assert df.index[0] == ("labeled-data", "endovis_seq1", "img0059.png")   # 1-based -> 0-based, 4 digits
    assert df[("du2018", "tool2_LeftClasperPoint", "x")].iloc[0] == 345.6
    assert np.isnan(df[("du2018", "tool1_HeadPoint", "x")].iloc[0])
    assert not any(b.startswith(("lhook", "grasper")) for b in bodyparts())
