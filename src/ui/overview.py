"""Overview helpers for the app: held-out reliability badges and an event list built from results.csv."""

from __future__ import annotations

import pandas as pd

# Held-out numbers measured in this project (see docs/). Pixel errors are on 854x480 frames.
BADGES = [
    ("Instrument masks", "Dice 0.85 / 0.84 (grasper / L-hook), unseen patients", "good"),
    ("Tissue (gallbladder model)", "mean Dice 0.735, liver 0.843, unseen patients", "good"),
    ("Tissue (other-surgery model)", "DSAD test Dice 0.846; gallbladder 0.697", "fair"),
    ("L-hook tip (DLC, clean test)", "median 6.5 px, 35/35 found, 2 unseen patients", "good"),
    ("Grasper tip (geometry)", "median 67 px on 5 videos - rough", "weak"),
    ("Bleeding detector", "F1 0.937, AUROC 0.942, unseen clips", "good"),
    ("CVS (safe to clip?)", "mAP 0.539, 40 unseen videos", "fair"),
    ("Distances in mm", "approx. from 5 mm shaft width, +/-30-50 %", "weak"),
    ("Risk score", "rule-based, NOT clinically validated", "weak"),
]
COLOR = {"good": "#1e8e3e", "fair": "#e37400", "weak": "#c5221f"}
RISK_ORDER = ["Unavailable", "Minimal", "Low", "Medium", "High"]
RISK_COLOR = ["#9aa0a6", "#34a853", "#a8c545", "#f9ab00", "#d93025"]


def _segments(t: pd.Series, flag: pd.Series, min_s: float):
    """[(start, end)] of runs where flag is True and that last at least min_s seconds."""
    out, start, prev = [], None, None
    for ti, f in zip(t, flag):
        if f and start is None:
            start = ti
        if not f and start is not None:
            if prev - start >= min_s:
                out.append((start, prev))
            start = None
        prev = ti
    if start is not None and prev - start >= min_s:
        out.append((start, prev))
    return out


def events(df: pd.DataFrame, min_s: float = 0.5) -> pd.DataFrame:
    """Time-ordered list of notable episodes: high/medium risk, bleeding, CVS achieved."""
    t = df["timestamp_s"]
    rows = []
    cat = df.get("risk_category", pd.Series(["Unavailable"] * len(df))).fillna("Unavailable")
    for name, mask in (("High risk", cat.eq("High")), ("Medium risk", cat.eq("Medium"))):
        rows += [(s, e, name) for s, e in _segments(t, mask, min_s)]
    if "bleeding_detected" in df:
        rows += [(s, e, "Bleeding flagged") for s, e in _segments(t, df["bleeding_detected"].fillna(False).astype(bool), min_s)]
    if "cvs_achieved" in df:
        rows += [(s, e, "CVS criteria all met") for s, e in _segments(t, df["cvs_achieved"].fillna(False).astype(bool), min_s)]
    ev = pd.DataFrame(rows, columns=["start_s", "end_s", "event"]).sort_values("start_s", kind="stable")
    ev["duration_s"] = (ev["end_s"].astype(float) - ev["start_s"].astype(float)).round(1)
    return ev.reset_index(drop=True)


def risk_strip(df: pd.DataFrame) -> str:
    """HTML colour strip of risk category over time (contiguous runs merged) with a legend."""
    t = df["timestamp_s"].astype(float).tolist()
    cat = df.get("risk_category", pd.Series(["Unavailable"] * len(df))).fillna("Unavailable").tolist()
    if not t:
        return ""
    total = max(t[-1] - t[0], 1e-6)
    col = dict(zip(RISK_ORDER, RISK_COLOR))
    segs, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or cat[i] != cat[start]:
            end = t[i] if i < len(t) else t[-1]
            segs.append((t[start], end, cat[start]))
            start = i
    bar = "".join(f"<div title='{c}: {s0:.1f}-{s1:.1f} s' style='flex:{max(s1 - s0, total / 2000):.4f};"
                  f"background:{col.get(c, '#9aa0a6')}'></div>" for s0, s1, c in segs)
    legend = " ".join(f"<span style='display:inline-block;width:10px;height:10px;background:{c};"
                      f"border-radius:2px;margin:0 3px 0 10px'></span>{n}" for n, c in zip(RISK_ORDER, RISK_COLOR))
    return (f"<div style='display:flex;height:22px;border-radius:4px;overflow:hidden'>{bar}</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.75em;opacity:0.7'>"
            f"<span>{t[0]:.0f} s</span><span>{t[-1]:.0f} s</span></div>"
            f"<div style='font-size:0.8em;margin-bottom:8px'>{legend}</div>")
