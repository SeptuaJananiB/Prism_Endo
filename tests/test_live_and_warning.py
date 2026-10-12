import numpy as np

from src.pipeline.overlay import draw_contact_warning
from src.utils.video_io import live_target


def test_live_target_parsing():
    assert live_target("camera:0") == 0
    assert live_target("2") == 2
    assert live_target("rtsp://10.0.0.5/stream") == "rtsp://10.0.0.5/stream"
    assert live_target("http://host/video.mjpg") == "http://host/video.mjpg"
    assert live_target("Videos/clip.mp4") is None
    assert live_target("camera:x") is None


def test_contact_warning_draws_red_border_and_label():
    img = np.zeros((120, 200, 3), np.uint8)
    out = draw_contact_warning(img.copy(), "grasper tip -> liver in 0.6 s", 0)
    assert out[2, 100, 2] > 200 and out[2, 100, 0] < 80   # red border (BGR)
    assert (out[100:110, 20:180] > 0).any()                 # label drawn near the bottom


def test_fast_live_tracks_tip_and_warns_on_approach():
    from src.api.live_fast import FastLive

    h, w = 240, 320
    tissue = np.zeros((h, w), np.uint8)
    tissue[:, 250:] = 1                                   # target tissue on the right
    raw = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    fl = FastLive(warn_s=1.0, min_approach_px_s=5.0)
    fl.keyframe(raw, {"grasper_tip_x": 100.0, "grasper_tip_y": 120.0, "grasper_tip_kp_status": "ok", "risk_category": "Low"},
                None, tissue, ["background", "liver"], ["liver"], ["grasper_tip"])
    info = {}
    for k in range(1, 12):                               # texture moves right 12 px per frame (tip approaching)
        shifted = np.roll(raw, 12 * k, axis=1)
        fl._last_t = (fl._last_t or 0)
        info = fl.frame(shifted)
    assert info["tips"] == 1
    tip = fl.tips["grasper_tip"]
    assert tip[0] > 200                                   # optical flow followed the motion
    out, n = fl.render()
    assert out is not None and out.shape == raw.shape
