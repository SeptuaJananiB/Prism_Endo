"""Mode 1: rule-based RESEARCH risk score (heuristic, not a probability).

The score is a transparent weighted combination of normalised features,
each in [0, 1]:

  f_prox    = 1 - d / d_ref           (clipped; 1 if the tip is inside the target)
  f_approach= v_towards / v_ref_a     (only motion TOWARDS the nearest target boundary)
  f_speed   = |v| / v_ref
  f_abrupt  = max(|a| / a_ref, turn_angle / 180 deg)

with every distance/speed expressed as a fraction of the image diagonal
(per second) so thresholds do not depend on video resolution. With tissue
available ("proximity" mode):

  score = w_prox*f_prox + g * (w_approach*f_approach + w_speed*f_speed + w_abrupt*f_abrupt)
  g     = clip(1 - d / (gate_mult * d_ref), 0, 1)   # motion only matters near tissue

Without tissue ("motion_only" mode, if allowed) the score is a motion
activity index: (w_speed*f_speed + w_abrupt*f_abrupt) / (w_speed + w_abrupt).
It says nothing about tissue and is labelled as such.

The score is exponentially smoothed over time and mapped to categories with
a dwell time (a new category must persist `dwell_s` seconds) plus an
immediate escalation when the smoothed score rises faster than
`rapid_rise_per_s`. All constants are configurable defaults chosen for
plausibility, NOT calibrated against injury or near-miss data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

CATEGORIES = ["Minimal", "Low", "Medium", "High"]
CONTRIBUTION_KEYS = ("proximity", "approach", "speed", "abrupt")


def _clip01(v: float) -> float:
    return 0.0 if v != v else max(0.0, min(1.0, v))  # NaN -> 0


@dataclass
class RiskOutput:
    keypoint: str
    mode: str                       # proximity | motion_only | disabled
    status: str                     # ok | partial | no_keypoint | low_confidence | disabled
    raw_score: float | None = None
    score: float | None = None      # smoothed
    category: str = "Unavailable"
    features: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)
    explanation: str = ""
    missing: list = field(default_factory=list)

    def as_dict(self, prefix: str = "") -> dict:
        d = asdict(self)
        out = {f"{prefix}{k}": d[k] for k in ("mode", "status", "raw_score", "score", "category", "explanation")}
        out[f"{prefix}missing"] = ";".join(self.missing)
        # Fixed set of columns; None where a term was not computed (e.g. motion-only mode).
        for k in CONTRIBUTION_KEYS:
            out[f"{prefix}contrib_{k}"] = self.contributions.get(k)
        return out


class _CategoryState:
    def __init__(self, thresholds: dict, dwell_s: float, rapid_rise_per_s: float):
        self.t = (thresholds["low"], thresholds["medium"], thresholds["high"])
        self.dwell_s = dwell_s
        self.rapid = rapid_rise_per_s
        self.current = "Unavailable"
        self.pending = None
        self.pending_since = None
        self.last = None  # (t, score)

    def raw_category(self, s: float) -> str:
        lo, me, hi = self.t
        return "Minimal" if s < lo else "Low" if s < me else "Medium" if s < hi else "High"

    def update(self, t: float, s: float) -> str:
        cand = self.raw_category(s)
        rising_fast = False
        if self.last is not None and t > self.last[0]:
            rising_fast = (s - self.last[1]) / (t - self.last[0]) > self.rapid
        self.last = (t, s)
        if self.current == "Unavailable":
            self.current = cand
        elif cand == self.current:
            self.pending = None
        elif rising_fast and CATEGORIES.index(cand) > CATEGORIES.index(self.current):
            self.current, self.pending = cand, None
        else:
            if cand != self.pending:
                self.pending, self.pending_since = cand, t
            if t - self.pending_since >= self.dwell_s:
                self.current, self.pending = cand, None
        return self.current

    def reset(self):
        self.current, self.pending, self.last = "Unavailable", None, None


class RuleBasedRiskScorer:
    """One instance per tracked keypoint (it keeps temporal state)."""

    def __init__(self, cfg: dict, keypoint: str):
        self.cfg = cfg
        self.kp = keypoint
        self.w = cfg["weights"]
        self.alpha = float(cfg.get("smoothing_alpha", 0.4))
        self.reset_after_s = float(cfg.get("reset_after_missing_s", 1.0))
        self.cat = _CategoryState(cfg["thresholds"], float(cfg.get("dwell_s", 0.2)),
                                  float(cfg.get("rapid_rise_per_s", 1.5)))
        self.smoothed = None
        self.last_valid_t = None

    def _features(self, motion, prox, diag: float) -> tuple[dict, list]:
        c, f, missing = self.cfg, {}, []
        speed = motion.speed_px_s if motion is not None else None
        if c.get("use_camera_compensated_speed") and getattr(motion, "speed_comp_px_s", None) is not None:
            speed = motion.speed_comp_px_s  # instrument motion relative to the scene
        if speed is None:
            missing.append("motion")
        f["speed_norm_per_s"] = None if speed is None else speed / diag
        f["f_speed"] = _clip01((speed or 0.0) / diag / c["speed_ref_frac_per_s"])
        acc = getattr(motion, "accel_px_s2", None)
        turn = getattr(motion, "turn_angle_deg", None)
        f["f_abrupt"] = max(_clip01((acc or 0.0) / diag / c["accel_ref_frac_per_s2"]),
                            _clip01((turn or 0.0) / 180.0))
        if prox is not None and prox.status == "ok":
            d = prox.distance_norm
            f["distance_norm"] = d
            f["target"] = prox.target
            f["f_prox"] = 1.0 if prox.inside else _clip01(1.0 - d / c["proximity_ref_frac"])
            gate = 1.0 if prox.inside else _clip01(1.0 - d / (c.get("gate_mult", 2.0) * c["proximity_ref_frac"]))
            f["gate"] = gate
            ap = prox.approach_speed_px_s
            f["f_approach"] = _clip01((ap or 0.0) / diag / c["approach_ref_frac_per_s"])
        return f, missing

    def update(self, t: float, keypoint, motion, prox_nearest, tissue_available: bool,
               image_diag: float) -> RiskOutput:
        mode = "proximity" if tissue_available else "motion_only"
        if not tissue_available and self.cfg.get("when_proximity_unavailable", "motion_only") == "disable":
            return RiskOutput(self.kp, "disabled", "disabled",
                              explanation="Risk disabled: tissue segmentation unavailable.",
                              missing=["tissue_model"])
        if keypoint is None or not keypoint.valid:
            status = "low_confidence" if keypoint is not None and keypoint.status == "low_confidence" \
                else "no_keypoint"
            if self.last_valid_t is not None and t - self.last_valid_t > self.reset_after_s:
                self.smoothed = None
                self.cat.reset()
            return RiskOutput(self.kp, mode, status, explanation=f"{self.kp}: {status.replace('_', ' ')}",
                              missing=["keypoint"])
        self.last_valid_t = t
        f, missing = self._features(motion, prox_nearest, image_diag)
        w = self.w
        contrib = {}
        if mode == "proximity":
            if "f_prox" in f:
                contrib["proximity"] = w["proximity"] * f["f_prox"]
                g = f["gate"]
                contrib["approach"] = g * w["approach"] * f["f_approach"]
                contrib["speed"] = g * w["speed"] * f["f_speed"]
                contrib["abrupt"] = g * w["abrupt"] * f["f_abrupt"]
            else:
                # Tissue model ran but no target class is visible in this frame.
                missing.append("target_tissue_not_visible")
                contrib = {"proximity": 0.0, "approach": 0.0, "speed": 0.0, "abrupt": 0.0}
            raw = sum(contrib.values())
        else:
            denom = w["speed"] + w["abrupt"]
            contrib["speed"] = w["speed"] * f["f_speed"] / denom
            contrib["abrupt"] = w["abrupt"] * f["f_abrupt"] / denom
            raw = sum(contrib.values())
            missing.append("tissue_model")
        raw = _clip01(raw)
        self.smoothed = raw if self.smoothed is None else self.alpha * raw + (1 - self.alpha) * self.smoothed
        category = self.cat.update(t, self.smoothed)
        out = RiskOutput(self.kp, mode, "partial" if missing else "ok", round(raw, 4), round(self.smoothed, 4),
                         category, f, {k: round(v, 4) for k, v in contrib.items()}, "", missing)
        out.explanation = self._explain(out, motion, prox_nearest)
        return out

    def _explain(self, out: RiskOutput, motion, prox) -> str:
        thr = float(self.cfg.get("explain_min_contribution", 0.05))
        parts = []
        f = out.features
        c = out.contributions
        if c.get("proximity", 0) >= thr and prox is not None:
            where = "inside" if prox.inside else f"{prox.distance_px:.0f}px from"
            parts.append(f"tip {where} {prox.target} (+{c['proximity']:.2f})")
        if c.get("approach", 0) >= thr and prox is not None:
            parts.append(f"moving towards {prox.target} at {prox.approach_speed_px_s:.0f}px/s (+{c['approach']:.2f})")
        if c.get("speed", 0) >= thr and motion is not None and motion.speed_px_s is not None:
            parts.append(f"fast motion {motion.speed_px_s:.0f}px/s (+{c['speed']:.2f})")
        if c.get("abrupt", 0) >= thr:
            parts.append(f"abrupt trajectory change (+{c['abrupt']:.2f})")
        head = f"{self.kp}: {out.category}"
        if out.mode == "motion_only":
            head += " [motion-only: no tissue info]"
        if "target_tissue_not_visible" in out.missing:
            head += " [no target tissue visible]"
        return head + (" - " + "; ".join(parts) if parts else " - no strong risk factors")


def combine(outputs: list[RiskOutput]) -> RiskOutput | None:
    """Frame-level risk = the keypoint with the highest smoothed score."""
    scored = [o for o in outputs if o.score is not None]
    if not scored:
        return None
    return max(scored, key=lambda o: o.score)
