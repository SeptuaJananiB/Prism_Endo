"""Plain-language explanation of an analysis with gpt-oss (OpenAI open-weight model) on an OpenAI-compatible API.

Only analysis NUMBERS are sent (risk timeline, events, bleeding, CVS, instruments, settings) - never video or frames.
Configure in a local .env file (never committed):

  GROQ_API_KEY=...                       # https://console.groq.com/keys
  PE_LLM_MODEL=openai/gpt-oss-120b       # optional; or openai/gpt-oss-20b
  PE_LLM_BASE_URL=https://api.groq.com/openai/v1   # optional; any OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM = """You explain the output of PRISM-Endo, a research tool that analyses laparoscopic surgery video.
You receive ONLY numbers computed by the tool (no images). Rules:
- Use only the data given. Never invent anatomy, steps, complications or numbers that are not in the data.
- Say "the analysis estimates" / "suggests"; these are model estimates, not clinical findings.
- Do not give medical advice or judge the surgeon.
- Distances are image pixels unless stated; the risk score is rule-based, not an injury probability.
- Mention reliability limits that the data reports (untested tissues, low-confidence tips).
- "tissue_segmentation_model_used" says which model was used, NOT which organ appears in the video.
- "proximity_targets" are the tissues the tips were measured against; only these were tracked for distance.
Write clear, short Markdown with exactly these sections:
## What happened in the video
(2-4 sentences: duration, which instruments were active and when, which tissue the tools worked near, overall flow)
## Key moments
(bullet list with times in m:ss, from the events and milestones)
## Risk and safety
(risk distribution, peaks and why, bleeding, Critical View of Safety)
## How far to trust this
(1-3 bullets)"""


def load_env() -> None:
    """Read KEY=VALUE lines from ROOT/.env into os.environ (without overriding real env vars)."""
    f = ROOT / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def config() -> dict:
    load_env()
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("PE_LLM_API_KEY")
    return {"configured": bool(key), "model": os.environ.get("PE_LLM_MODEL", DEFAULT_MODEL),
            "base_url": os.environ.get("PE_LLM_BASE_URL", DEFAULT_BASE).rstrip("/"), "key": key}


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _runs(t: np.ndarray, flag: np.ndarray, min_s: float, gap_s: float = 0.4) -> list[tuple[float, float]]:
    out, st, last = [], None, None
    for i, f in enumerate(flag):
        if f:
            st = i if st is None else st
            last = i
        elif st is not None and t[i] - t[last] > gap_s:
            if t[last] - t[st] >= min_s:
                out.append((float(t[st]), float(t[last])))
            st = None
    if st is not None and t[last] - t[st] >= min_s:
        out.append((float(t[st]), float(t[last])))
    return out


def build_context(run_dir: Path) -> dict:
    """Compact, numbers-only description of one analysis."""
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(run_dir / "results.csv")
    t = df["timestamp_s"].to_numpy(float)
    n = len(df)
    cat = df.get("risk_category", pd.Series(["Unavailable"] * n)).fillna("Unavailable")
    truthy = lambda c: df[c].fillna(False).astype(bool).to_numpy() if c in df else np.zeros(n, bool)  # noqa: E731
    ctx: dict = {
        "duration_s": round(float(t[-1]) if n else 0.0, 1), "frames": n,
        "tissue_segmentation_model_used": ("model trained on other abdominal surgeries (DSAD)" if "dsad" in str(s.get("tissue_checkpoint", ""))
                                           else "model trained on gallbladder surgery videos (CholecSeg8k)"),
        "tip_source": s.get("keypoint_source"),
        "proximity_targets": {k: v.get("reliability") for k, v in (s.get("proximity_targets") or {}).items()},
        "risk_share_pct": {k: round(100 * float((cat == k).mean()), 1) for k in ("Minimal", "Low", "Medium", "High", "Unavailable")},
    }
    if "risk_score" in df and df["risk_score"].notna().any():
        i = int(df["risk_score"].idxmax())
        expl = df["risk_explanation"].iloc[i] if "risk_explanation" in df else ""
        ctx["peak_risk"] = {"time": _mmss(t[i]), "score": round(float(df["risk_score"].iloc[i]), 2), "reason": str(expl)[:220]}
    hi = _runs(t, cat.isin(["High"]).to_numpy(), 0.1)
    med = _runs(t, cat.isin(["Medium", "High"]).to_numpy(), 2.0)
    ctx["high_risk_episodes"] = [f"{_mmss(a)}-{_mmss(b)}" for a, b in hi[:12]]
    ctx["sustained_medium_or_high_episodes"] = [f"{_mmss(a)}-{_mmss(b)}" for a, b in med[:12]]
    b = truthy("bleeding_detected")
    ctx["bleeding"] = {"frames_flagged": int(b.sum()), "episodes": [f"{_mmss(x)}-{_mmss(y)}" for x, y in _runs(t, b, 0.3)[:10]]}
    if "cvs_achieved" in df:
        c = truthy("cvs_achieved")
        ctx["critical_view_of_safety"] = {"frames_met_pct": round(100 * float(c.mean()), 1),
                                          "first_met": _mmss(t[np.argmax(c)]) if c.any() else None}
    for name, col in (("grasper", "inst_grasper_frac"), ("l_hook", "inst_l_hook_frac")):
        if col in df:
            v = df[col].fillna(0).to_numpy() > 0.002
            ctx.setdefault("instruments_visible", {})[name] = {
                "pct_of_video": round(100 * float(v.mean()), 1),
                "periods": [f"{_mmss(a)}-{_mmss(b2)}" for a, b2 in _runs(t, v, 1.0)[:8]]}
    dist = [c for c in df.columns if c.endswith("_prox_distance_px")]
    if dist:
        d = df[dist].min(axis=1)
        if d.notna().any():
            i = int(d.idxmin())
            ctx["closest_approach"] = {"time": _mmss(t[i]), "distance_px": round(float(d.iloc[i]), 1)}
            ctx["time_tip_within_60px_of_target_pct"] = round(100 * float((d <= 60).mean()), 1)
    low = [c for c in df.columns if c.endswith("_kp_status")]
    if low:
        ctx["low_confidence_tip_frames_pct"] = round(100 * float(df[low].isin(["low_confidence"]).any(axis=1).mean()), 1)
    ctx["tool_warnings"] = [w[:200] for w in (s.get("warnings") or [])][:5]
    return ctx


def explain(run_dir: Path, refresh: bool = False) -> dict:
    """Return {text, model, created, cached}. Cached in run_dir/explanation.json (invalidated when results change)."""
    cfg = config()
    if not cfg["configured"]:
        raise RuntimeError("No API key: add GROQ_API_KEY=... to the .env file in the project folder, then restart the app.")
    cache = run_dir / "explanation.json"
    stamp = (run_dir / "results.csv").stat().st_mtime
    if cache.is_file() and not refresh:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("results_mtime") == stamp and c.get("model") == cfg["model"]:
            return {**c, "cached": True}
    ctx = build_context(run_dir)
    body = {"model": cfg["model"], "temperature": 0.2, "max_tokens": 900,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Analysis data (JSON):\n" + json.dumps(ctx, indent=1)}]}
    if "gpt-oss" in cfg["model"]:
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(f"{cfg['base_url']}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json",
                                          "User-Agent": "PRISM-Endo"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Model service error {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach the model service: {e.reason}") from None
    text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    # models sometimes bold their headings ("**## Title**"): normalise to plain Markdown headings
    import re
    text = re.sub(r"^\*\*\s*(#{1,4}\s+.*?)\s*\*\*\s*$", r"\1", text, flags=re.M)
    if not text:
        raise RuntimeError("The model returned an empty answer; try again.")
    res = {"text": text, "model": cfg["model"], "created": time.time(), "results_mtime": stamp, "context": ctx}
    cache.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return {**res, "cached": False}
