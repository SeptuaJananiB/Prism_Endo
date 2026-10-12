import numpy as np

from src.proximity.scale import mm_per_px


def test_shaft_width_gives_scale():
    m = np.zeros((300, 600), np.uint8)
    m[130:170, 0:400] = 1          # 40 px wide shaft, tip at x=399
    s = mm_per_px(m, (399, 150), shaft_mm=5.0)
    assert s is not None and abs(s - 5.0 / 40) < 0.02


def test_implausible_or_missing_gives_none():
    assert mm_per_px(np.zeros((100, 100), np.uint8), (50, 50)) is None
    m = np.zeros((300, 600), np.uint8)
    m[100:104, 0:300] = 1          # 4 px wide: below plausible shaft width
    assert mm_per_px(m, (299, 102)) is None
    assert mm_per_px(m, (float("nan"), 0)) is None
