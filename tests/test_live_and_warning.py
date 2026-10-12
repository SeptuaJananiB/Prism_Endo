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
    fl.keyframe(raw, {"grasper_tip_x": 100.0, "grasper_tip_y": 120.0, "grasper_tip_kp_status": "ok", "risk_category": "High"},
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
    snap = fl.snapshot()                                  # live charts / readouts / events for the browser
    assert snap["series"] and "grasper_tip" in snap["series"][-1]["tips"]
    assert snap["series"][-1]["tips"]["grasper_tip"]["target"] == "liver"
    assert any(e["kind"] == "Contact warning" for e in snap["events"])
    assert fl.snapshot(since=snap["series"][-1]["frame"])["series"] == []
    assert len(snap["series"]) == 11 and snap["series"][-1]["frame"] == 11   # one sample per tracked frame


def test_fast_live_risk_is_computed_for_every_frame():
    """Risk comes from the tracked tip on each frame (not repeated from the keyframe) and rises as the tip nears tissue."""
    from src.api.live_fast import FastLive

    h, w = 240, 320
    tissue = np.zeros((h, w), np.uint8)
    tissue[:, 250:] = 1
    raw = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    fl = FastLive(warn_s=1.0, min_approach_px_s=5.0)
    fl.keyframe(raw, {"grasper_tip_x": 60.0, "grasper_tip_y": 120.0, "grasper_tip_kp_status": "ok", "risk_category": "Minimal"},
                None, tissue, ["background", "liver"], ["liver"], ["grasper_tip"])
    scores = []
    for k in range(1, 19):
        info = fl.frame(np.roll(raw, 12 * k, axis=1))
        scores.append(info["risk_score"])
    series = fl.snapshot()["series"]
    assert len(series) == 18 and all(x["score"] is not None for x in series)   # a risk value on every frame
    assert scores[-1] > scores[0]                                              # closer to tissue -> higher risk
    assert series[-1]["risk"] in ("Medium", "High") and fl.risk != "Minimal"   # not the keyframe's category
    assert "liver" in fl.snapshot()["info"]["explanation"]


def test_fast_live_no_tip_means_no_risk():
    from src.api.live_fast import FastLive

    fl = FastLive()
    img = np.zeros((120, 160, 3), np.uint8)
    fl.keyframe(img, {"risk_category": "Unavailable"}, None, None, [], [], ["grasper_tip"])
    info = fl.frame(img)
    assert info["risk"] == "Minimal" and info["risk_score"] == 0.0
    assert fl.snapshot()["series"][-1]["score"] == 0.0
