import numpy as np

from src.safety import dissection_zones as dz

NAMES = ["background", "liver", "fat", "gallbladder", "cystic_duct", "gastrointestinal_tract"]


def _scene():
    lab = np.zeros((200, 300), np.int32)
    lab[:, :] = NAMES.index("fat")
    lab[60:140, 20:120] = NAMES.index("gallbladder")
    lab[90:100, 200:260] = NAMES.index("cystic_duct")
    return lab


def test_go_is_fat_next_to_gallbladder_and_nogo_is_deep_or_near_danger():
    z = dz.compute(_scene(), NAMES)
    assert z.available
    assert z.zone_at(125, 100) == "go"            # fat just beside the gallbladder
    assert z.zone_at(230, 95) == "no_go"          # on a danger structure (cystic duct)
    assert z.zone_at(290, 10) == "no_go"          # deep tissue far from the gallbladder
    assert z.zone_at(70, 100) == ""               # on the gallbladder itself: neither
    assert not (z.go & z.nogo).any()


def test_no_zones_without_visible_gallbladder_or_when_disabled():
    lab = np.full((100, 100), NAMES.index("fat"), np.int32)
    assert not dz.compute(lab, NAMES).available
    assert not dz.compute(_scene(), NAMES, {"enabled": False}).available
    assert not dz.compute(None, NAMES).available


def test_draw_adds_overlay():
    img = np.zeros((200, 300, 3), np.uint8)
    out = dz.draw(img, dz.compute(_scene(), NAMES))
    assert out.shape == img.shape and out.any()
