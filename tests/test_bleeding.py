import numpy as np

from src.safety.bleeding import draw_bleeding


def test_draw_bleeding_only_when_detected():
    img = np.zeros((100, 120, 3), np.uint8)
    mask = np.zeros((100, 120), bool)
    mask[30:60, 40:80] = True
    assert draw_bleeding(img.copy(), mask, False).sum() == 0
    out = draw_bleeding(img.copy(), mask, True)
    assert out[:, :, 2].sum() > 0 and out[:, :, 0].sum() == 0  # red outline only


def test_draw_bleeding_handles_missing_mask():
    img = np.zeros((10, 10, 3), np.uint8)
    assert draw_bleeding(img, None, True) is img
