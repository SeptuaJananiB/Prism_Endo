"""The click tool's writer must produce files DeepLabCut and our status/IO code can read."""
import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import yaml

from src.tracking.dlc_io import DLCPredictions
from src.tracking.dlc_workflow import label_status


def load_tool(monkeypatch, tmp_path):
    fake = types.ModuleType("streamlit_image_coordinates")
    fake.streamlit_image_coordinates = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "streamlit_image_coordinates", fake)
    spec = importlib.util.spec_from_file_location("label_tips_mod", "src/ui/label_tips.py")
    mod = importlib.util.module_from_spec(spec)
    src = open("src/ui/label_tips.py", encoding="utf-8").read().split("kind = st.sidebar.radio")[0]
    exec(compile(src, "label_tips", "exec"), mod.__dict__)
    return mod


def test_writer_roundtrip(monkeypatch, tmp_path):
    tool = load_tool(monkeypatch, tmp_path)
    proj = tmp_path / "prism-tips-train-x"
    (proj / "labeled-data" / "clipA").mkdir(parents=True)
    bps = ["lhook_tip", "grasper_tip", "grasper_tip_2"]
    (proj / "config.yaml").write_text(yaml.safe_dump({"bodyparts": bps, "scorer": "annotator"}))
    for n in ("img05.png", "img10.png"):
        (proj / "labeled-data" / "clipA" / n).write_bytes(b"")
    # TEST FIXTURE values (temporary directory), not real annotations.
    table = {"img05.png": {"lhook_tip": (100.0, 50.0), "grasper_tip": (np.nan, np.nan)},
             "img10.png": {b: (np.nan, np.nan) for b in bps}}
    tool.write_labels(proj, "clipA", "annotator", bps, table)
    st = label_status(proj / "config.yaml")
    assert st["frames_labelled"] == 1 and st["per_bodypart"]["lhook_tip"] == 1
    pred = DLCPredictions(proj / "labeled-data" / "clipA" / "CollectedData_annotator.csv")
    assert pred.get(5, "lhook_tip") == (100.0, 50.0, 1.0)
    back = tool.read_labels(proj, "clipA", "annotator", bps)
    assert back["img05.png"]["lhook_tip"] == (100.0, 50.0) and np.isnan(back["img10.png"]["grasper_tip"][0])
