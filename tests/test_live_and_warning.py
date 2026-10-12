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
