import pandas as pd

from src.ui.overview import events


def test_events_finds_episodes_and_skips_blips():
    t = [i * 0.1 for i in range(30)]
    cat = ["Low"] * 5 + ["High"] * 10 + ["Low"] * 10 + ["High"] * 2 + ["Low"] * 3
    bleed = [False] * 20 + [True] * 10
    ev = events(pd.DataFrame({"timestamp_s": t, "risk_category": cat, "bleeding_detected": bleed}), min_s=0.5)
    assert list(ev["event"]) == ["High risk", "Bleeding flagged"]  # 0.1 s High blip dropped
    assert ev.loc[0, "start_s"] == 0.5 and abs(ev.loc[0, "end_s"] - 1.4) < 1e-9


def test_events_empty_without_columns():
    assert events(pd.DataFrame({"timestamp_s": [0.0, 0.1]})).empty


def test_risk_strip_merges_runs():
    from src.ui.overview import risk_strip
    html = risk_strip(pd.DataFrame({"timestamp_s": [0, 1, 2, 3], "risk_category": ["Low", "Low", "High", "High"]}))
    assert html.count("title='Low") == 1 and html.count("title='High") == 1
