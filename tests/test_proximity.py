import math

import numpy as np
import pytest

from src.proximity.proximity import ProximityEstimator, nearest, proximity_to_mask


def square_mask(h=200, w=300, x0=100, y0=50, x1=200, y1=150):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_outside_distance_and_closest_point():
    m = square_mask()
    r = proximity_to_mask(m, 50.0, 100.0, "liver")  # 50 px left of the left edge (x=100)
    assert r.status == "ok" and r.inside is False
    assert r.distance_px == pytest.approx(50.0)
    assert r.signed_distance_px == pytest.approx(50.0)
    assert r.closest_x == pytest.approx(100.0) and r.closest_y == pytest.approx(100.0)
    assert r.distance_norm == pytest.approx(50.0 / math.hypot(300, 200))


def test_inside_has_negative_signed_distance():
    r = proximity_to_mask(square_mask(), 110.0, 100.0, "liver")
    assert r.inside is True
    assert r.distance_px == pytest.approx(10.0)
    assert r.signed_distance_px == pytest.approx(-10.0)


def test_approach_speed_sign():
    m = square_mask()
    towards = proximity_to_mask(m, 50.0, 100.0, "liver", velocity=(30.0, 0.0))
    away = proximity_to_mask(m, 50.0, 100.0, "liver", velocity=(-30.0, 0.0))
    sideways = proximity_to_mask(m, 50.0, 100.0, "liver", velocity=(0.0, 30.0))
    assert towards.approach_speed_px_s == pytest.approx(30.0)
    assert away.approach_speed_px_s == pytest.approx(-30.0)
    assert sideways.approach_speed_px_s == pytest.approx(0.0, abs=1e-9)


def test_absent_and_empty_masks():
    empty = np.zeros((100, 100), bool)
    assert proximity_to_mask(empty, 5, 5).status == "absent"
    tiny = np.zeros((100, 100), bool)
    tiny[10, 10] = True
    assert proximity_to_mask(tiny, 5, 5, min_area=50).status == "absent"


def test_image_border_is_not_a_tissue_boundary():
    # Region touches the left image border; the point is near that border but
    # far from the region's real (right) edge at x=150.
    m = np.zeros((100, 300), bool)
    m[:, :150] = True
    r = proximity_to_mask(m, 2.0, 50.0, "fat")
    assert r.inside is True
    assert r.distance_px == pytest.approx(148.0, abs=1.0)


def test_estimator_unavailable_and_invalid_keypoint():
    est = ProximityEstimator(["liver", "cystic_duct"])
    res = est(None, [], 10, 10)
    assert [r.status for r in res] == ["unavailable", "unavailable"]
    labels = np.zeros((100, 100), np.uint8)
    labels[40:60, 40:60] = 1
    res = est(labels, ["background", "liver", "cystic_duct"], None, None)
    assert all(r.status == "no_keypoint" for r in res)
    res = est(labels, ["background", "liver", "cystic_duct"], 10.0, 50.0)
    assert res[0].status == "ok" and res[1].status == "absent"
    assert nearest(res).target == "liver"


def test_keypoint_outside_image_is_handled():
    r = proximity_to_mask(square_mask(), -20.0, 100.0, "liver")
    assert r.status == "ok" and r.inside is False
    assert r.distance_px == pytest.approx(120.0)


def test_fov_border_is_not_a_tissue_boundary():
    # Liver fills the right half up to a black FOV border (class 0) on the far right.
    labels = np.full((100, 300), 1, np.uint8)
    labels[:, :150] = 2          # other tissue on the left
    labels[:, 290:] = 0          # black field-of-view border
    est = ProximityEstimator(["liver"], border_class=0)
    r = est(labels, ["background", "liver", "other"], 285.0, 50.0)[0]
    assert r.inside is True
    # Nearest real boundary is the liver/other edge at x~150, not the FOV border at x~290.
    assert r.distance_px == pytest.approx(135.0, abs=2.0)
    no_excl = ProximityEstimator(["liver"], border_class=None)
    assert no_excl(labels, ["background", "liver", "other"], 285.0, 50.0)[0].distance_px < 10


def test_estimator_reuses_boundaries_within_a_frame():
    labels = np.zeros((100, 100), np.uint8)
    labels[40:60, 40:60] = 1
    est = ProximityEstimator(["liver"])
    a = est(labels, ["background", "liver"], 10.0, 50.0)[0]
    prepared = est._prepared
    b = est(labels, ["background", "liver"], 90.0, 50.0)[0]
    assert est._prepared is prepared and a.distance_px == pytest.approx(30.0) and b.distance_px == pytest.approx(31.0)
    # A new label map (even with identical content) must trigger a fresh extraction.
    moved = np.zeros_like(labels)
    moved[40:60, 70:90] = 1
    c = est(moved, ["background", "liver"], 80.0, 50.0)[0]
    assert c.inside is True
