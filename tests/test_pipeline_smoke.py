"""Pipeline smoke tests on a synthetic clip with stand-in segmenters.

The stand-ins threshold the synthetic colours; they only exist to exercise
the data flow (keypoints -> motion -> proximity -> risk -> outputs).
"""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.make_synthetic_video import main as make_video
from src.pipeline.pipeline import PrismPipeline
from src.utils.config import load_config, resolve_path


class FakeInstrument:
    class_names = ["background", "grasper", "l_hook"]

    def predict(self, bgr):
        b, g, r = [bgr[..., i].astype(int) for i in range(3)]
        grey = (abs(b - g) < 12) & (abs(g - r) < 12) & (b > 110)
        return np.where(grey, 2, 0).astype(np.uint8)  # everything grey = l_hook


class FakeTissue:
    available = True
    class_names = ["background", "liver", "cystic_duct", "gallbladder"]
    ignore_classes = set()
    message = "fake"
    target_info = {}
    warnings = []

    def predict(self, bgr):
        b, g, r = [bgr[..., i].astype(int) for i in range(3)]
        yellow = (g > 170) & (r > 190) & (b < 100)
        return np.where(yellow, 1, 0).astype(np.uint8)  # synthetic target region = "liver"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    make_video(["--out", str(d / "synthetic.mp4"), "--frames", "50"])
    return d


def base_cfg(**over):
    cfg = load_config(resolve_path("configs/pipeline.yaml"))
    cfg["instrument"]["checkpoint"] = None
    cfg["tissue"]["checkpoint"] = None
    cfg["output"]["browser_playable"] = False
    cfg["output"]["write_original_video"] = False
    cfg["keypoints"]["source"] = "mask_tip"
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def test_full_flow_with_tissue(synthetic, tmp_path):
    pipe = PrismPipeline(base_cfg())
    pipe.inst, pipe.tissue = FakeInstrument(), FakeTissue()
    summary = pipe.run(synthetic / "synthetic.mp4", tmp_path / "run")
    df = pd.read_csv(tmp_path / "run" / "results.csv")
    assert summary["frames_processed"] == 50 and len(df) == 50
    assert (tmp_path / "run" / "annotated.mp4").stat().st_size > 0
    truth = pd.read_csv(synthetic / "synthetic_tip_truth.csv")
    ok = df["lhook_tip_kp_status"] == "ok"
    assert ok.mean() > 0.9
    err = np.hypot(df.loc[ok, "lhook_tip_x"] - truth.loc[ok, "x"], df.loc[ok, "lhook_tip_y"] - truth.loc[ok, "y"])
    assert np.median(err) < 15  # heuristic tip lands on the drawn tip of this simple shape
    # Grasper is never drawn -> must be reported missing, never invented.
    assert (df["grasper_tip_kp_status"] == "missing").all()
    # Proximity is measured; the tip approaches the target so distance should drop.
    d = df["lhook_tip_prox_distance_px"].dropna()
    assert len(d) > 30 and d.iloc[-10:].min() < d.iloc[:10].max()
    assert set(df["risk_mode"].dropna()) <= {"proximity"}
    assert df["risk_score"].dropna().between(0, 1).all()
    assert df["lhook_tip_speed_px_s"].dropna().gt(0).any()
    json.loads((tmp_path / "run" / "summary.json").read_text())


def test_instrument_only_mode_reports_unavailable_proximity(synthetic, tmp_path):
    pipe = PrismPipeline(base_cfg())
    pipe.inst = FakeInstrument()
    summary = pipe.run(synthetic / "synthetic.mp4", tmp_path / "run")
    df = pd.read_csv(tmp_path / "run" / "results.csv")
    assert not summary["tissue_model"]
    assert any("instrument-only" in w for w in summary["warnings"])
    assert "lhook_tip_prox_distance_px" not in df.columns or df["lhook_tip_prox_distance_px"].isna().all()
    assert set(df["lhook_tip_prox_target"].dropna()) <= {"unavailable", "no_keypoint"}
    assert set(df["risk_mode"].dropna()) <= {"motion_only", "unavailable"}


def test_risk_can_be_disabled_without_tissue(synthetic, tmp_path):
    pipe = PrismPipeline(base_cfg(risk={"when_proximity_unavailable": "disable"}))
    pipe.inst = FakeInstrument()
    pipe.run(synthetic / "synthetic.mp4", tmp_path / "run")
    df = pd.read_csv(tmp_path / "run" / "results.csv")
    assert df["risk_score"].isna().all()
    assert (df["risk_category"] == "Unavailable").all()


def test_no_models_runs_and_warns(synthetic, tmp_path):
    cfg = base_cfg()
    cfg["keypoints"]["source"] = "auto"
    pipe = PrismPipeline(cfg)
    summary = pipe.run(synthetic / "synthetic.mp4", tmp_path / "run")
    assert summary["keypoint_source"] == "none"
    assert len(summary["warnings"]) >= 2


def test_image_folder_input_uses_filename_timestamps(tmp_path):
    folder = tmp_path / "clip"
    folder.mkdir()
    img = np.full((60, 80, 3), 100, np.uint8)
    for n in (100, 101, 102, 150):  # gap of 48 frames before the last one
        cv2.imwrite(str(folder / f"frame_{n}_endo.png"), img)
    pipe = PrismPipeline(base_cfg())
    pipe.inst = FakeInstrument()
    pipe.run(folder, tmp_path / "run")
    df = pd.read_csv(tmp_path / "run" / "results.csv")
    assert df["timestamp_s"].tolist() == pytest.approx([0.0, 0.04, 0.08, 2.0])


class LateInstrument(FakeInstrument):
    """No instrument for the first frames, then the normal stand-in."""

    def __init__(self, blank=5):
        self.calls, self.blank = 0, blank

    def predict(self, bgr):
        self.calls += 1
        out = super().predict(bgr)
        return out * 0 if self.calls <= self.blank else out


def test_csv_schema_does_not_depend_on_first_frame(synthetic, tmp_path):
    # Regression: columns must not be dropped when frame 0 has no keypoint/proximity.
    pipe = PrismPipeline(base_cfg())
    pipe.inst, pipe.tissue = LateInstrument(), FakeTissue()
    pipe.run(synthetic / "synthetic.mp4", tmp_path / "run")
    df = pd.read_csv(tmp_path / "run" / "results.csv")
    assert df.loc[:4, "lhook_tip_kp_status"].eq("missing").all()
    assert df["lhook_tip_prox_distance_px"].notna().sum() > 30
    assert df["lhook_tip_risk_contrib_proximity"].notna().sum() > 30
