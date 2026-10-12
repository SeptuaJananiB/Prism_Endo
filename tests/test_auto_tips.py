import numpy as np

from src.tracking.auto_tips import _geodesic_bfs, auto_tips, geodesic_distance, tip_from_component
from src.tracking.mask_tip import fov_edge_distance
from src.tracking.sources import KeypointSpec, MaskTipKeypointSource


def _bar(h=200, w=400, y0=90, y1=110, x1=300):
    """Horizontal instrument entering from the left image edge, ending at x1."""
    m = np.zeros((h, w), bool)
    m[y0:y1, 0:x1] = True
    return m


def test_geodesic_half_res_matches_full_res_roughly():
    comp = _bar()
    seeds = np.zeros_like(comp)
    seeds[:, 0] = True
    full, half = geodesic_distance(comp, seeds, 1), geodesic_distance(comp, seeds, 2)
    assert full.max() == 299
    assert abs(int(half.max()) - 299) <= 3
    assert (half[~comp] == -1).all()


def test_geodesic_follows_a_bend():
    comp = np.zeros((200, 200), bool)
    comp[90:100, 0:150] = True      # shaft
    comp[20:100, 140:150] = True    # bend upwards (L shape)
    seeds = np.zeros_like(comp)
    seeds[:, 0] = True
    d = _geodesic_bfs(comp, seeds)
    assert d[20, 145] > d[95, 149]  # the bent end is geodesically farthest


def test_far_end_and_hinge_positions():
    comp = _bar()
    de = fov_edge_distance(np.ones(comp.shape, bool))
    far = tip_from_component(comp, de, "far_end")
    hinge = tip_from_component(comp, de, "hinge", jaw_px=90)
    assert abs(far["x"] - 297) < 6 and abs(far["y"] - 99.5) < 2
    assert abs(hinge["x"] - (far["x"] - 90)) < 8
    assert 0 <= far["confidence"] <= 1


def test_absent_instrument_gives_missing_not_coordinates():
    lab = np.zeros((200, 400), np.uint8)
    out = auto_tips(lab, np.ones(lab.shape, bool))
    assert out == {"lhook_tip": None, "grasper_tip": None, "grasper_tip_2": None}


def test_two_graspers_ordered_left_entry_first():
    lab = np.zeros((300, 600), np.uint8)
    lab[50:70, 0:250] = 31           # enters from the left
    lab[200:220, 350:600] = 31       # enters from the right
    out = auto_tips(lab, np.ones(lab.shape, bool), "hinge", jaw_px=90)
    assert out["grasper_tip"]["x"] < out["grasper_tip_2"]["x"]


def test_mask_tip_source_uses_configured_hinge():
    specs = [KeypointSpec("grasper_tip", "grasper")]
    frame = np.full((200, 400, 3), 120, np.uint8)
    labels = np.zeros((200, 400), np.uint8)
    labels[_bar()] = 1
    old = MaskTipKeypointSource(specs, ["background", "grasper"], min_confidence=0.0)(frame, 0, labels)[0]
    new = MaskTipKeypointSource(specs, ["background", "grasper"], min_confidence=0.0,
                                landmarks={"grasper": {"landmark": "hinge", "jaw_px": 90}})(frame, 0, labels)[0]
    assert old.x > 280 and abs(new.x - (old.x - 90)) < 10


def test_short_instrument_falls_back_to_far_end():
    specs = [KeypointSpec("grasper_tip", "grasper")]
    frame = np.full((200, 400, 3), 120, np.uint8)
    labels = np.zeros((200, 400), np.uint8)
    labels[_bar(x1=60)] = 1   # only 60 px visible: shorter than the 90 px jaw
    kp = MaskTipKeypointSource(specs, ["background", "grasper"], min_confidence=0.0,
                               landmarks={"grasper": {"landmark": "hinge", "jaw_px": 90}})(frame, 0, labels)[0]
    assert kp.extra["method"].startswith("far_end_fallback") and kp.x > 50
