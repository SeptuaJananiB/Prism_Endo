import numpy as np

from src.tracking.keypoints import Keypoint, KeypointSpec
from src.tracking.sources import HybridKeypointSource, MaskTipKeypointSource


class FakeDLC:
    def __init__(self, x, y, conf):
        self.kp = (x, y, conf)

    def __call__(self, frame, idx, labels=None):
        x, y, c = self.kp
        return [Keypoint("grasper_tip", x, y, c, "dlc", status="ok" if c >= 0.6 else "low_confidence")]


def _setup():
    specs = [KeypointSpec("grasper_tip", "grasper")]
    frame = np.full((200, 400, 3), 120, np.uint8)
    labels = np.zeros((200, 400), np.uint8)
    labels[90:110, 0:300] = 1  # grasper from the left edge to x=300
    return MaskTipKeypointSource(specs, ["background", "grasper"], min_confidence=0.0), frame, labels


def test_dlc_point_on_instrument_is_used():
    mask_src, frame, labels = _setup()
    kp = HybridKeypointSource(mask_src, FakeDLC(250, 100, 0.2))(frame, 0, labels)[0]
    assert (kp.x, kp.y) == (250, 100) and kp.status == "ok" and kp.extra["method"] == "dlc_on_mask"


def test_dlc_point_on_tissue_falls_back_to_mask_tip():
    mask_src, frame, labels = _setup()
    kp = HybridKeypointSource(mask_src, FakeDLC(200, 30, 0.3))(frame, 0, labels)[0]  # 60 px above the instrument
    assert kp.x > 280 and kp.extra["method"].startswith("mask_tip_fallback")


def test_confident_dlc_point_is_used_even_off_mask():
    mask_src, frame, labels = _setup()
    kp = HybridKeypointSource(mask_src, FakeDLC(200, 30, 0.9))(frame, 0, labels)[0]
    assert (kp.x, kp.y) == (200, 30) and kp.extra["method"] == "dlc"


def test_slots_not_listed_use_mask_tip():
    mask_src, frame, labels = _setup()
    kp = HybridKeypointSource(mask_src, FakeDLC(250, 100, 0.9), dlc_slots=["lhook_tip"])(frame, 0, labels)[0]
    assert kp.x > 280
