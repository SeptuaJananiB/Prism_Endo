import json

import pytest

from src.tissue import tissue_segmenter
from src.tissue.support import SUPPORT_FILE, reliability

SUPPORT = {"splits": {"train": {"frames_with_class": {"liver": 880, "cystic_duct": 239, "hepatic_vein": 0}},
                      "test": {"frames_with_class": {"liver": 320, "cystic_duct": 0, "hepatic_vein": 0}}}}
METRICS = {"per_class": {"liver": {"dice": 0.93}}}


def test_reliability_levels():
    assert reliability(SUPPORT, METRICS, "liver")["reliability"] == "evaluated"
    assert reliability(SUPPORT, METRICS, "liver")["test_dice"] == 0.93
    assert reliability(SUPPORT, METRICS, "cystic_duct")["reliability"] == "unevaluated"
    assert reliability(SUPPORT, METRICS, "hepatic_vein")["reliability"] == "untrained"
    assert reliability(None, None, "liver")["reliability"] == "unknown"


class _FakePredictor:
    class_names = ["background", "liver", "cystic_duct", "hepatic_vein"]


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    path.write_bytes(b"x")
    (tmp_path / SUPPORT_FILE).write_text(json.dumps(SUPPORT))
    (tmp_path / "eval_test").mkdir()
    (tmp_path / "eval_test" / "metrics.json").write_text(json.dumps(METRICS))
    monkeypatch.setattr(tissue_segmenter.SegmentationPredictor, "from_checkpoint",
                        classmethod(lambda cls, p, d="cpu": _FakePredictor()))
    return path


def test_untrained_target_is_refused(ckpt):
    with pytest.raises(ValueError, match="hepatic_vein"):
        tissue_segmenter.TissueSegmenter(ckpt, target_classes=["liver", "hepatic_vein"])
    ts = tissue_segmenter.TissueSegmenter(ckpt, target_classes=["hepatic_vein"], allow_untrained_targets=True)
    assert any("untrained" in w for w in ts.warnings)


def test_unevaluated_target_is_warned(ckpt):
    ts = tissue_segmenter.TissueSegmenter(ckpt, target_classes=["liver", "cystic_duct"])
    assert ts.target_info["liver"]["reliability"] == "evaluated"
    assert ts.target_info["cystic_duct"]["reliability"] == "unevaluated"
    assert len(ts.warnings) == 1 and "cystic_duct" in ts.warnings[0]


def test_insufficient_and_failed_levels():
    sup = {"splits": {"train": {"frames_with_class": {"a": 500, "b": 500}},
                      "test": {"frames_with_class": {"a": 2, "b": 300}}}}
    met = {"per_class": {"a": {"dice": 0.0}, "b": {"dice": 0.2}}}
    assert reliability(sup, met, "a")["reliability"] == "insufficient_test_data"
    assert reliability(sup, met, "b")["reliability"] == "failed_on_heldout"
