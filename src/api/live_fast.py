"""Frame-rate live layer: keyframe analysis + per-frame tip tracking (Lucas-Kanade optical flow).

The full pipeline (segmentation, tips, tissue, bleeding, CVS) runs on keyframes as fast as the GPU allows. Between
keyframes, EVERY camera frame is processed here in a few milliseconds:
  * instrument tips are tracked with pyramidal Lucas-Kanade optical flow and re-anchored at each keyframe;
  * distance to each target tissue is read from distance maps computed once per keyframe;
  * speed, approach speed and time to contact are updated per frame (EMA smoothed);
  * the frame is drawn with the latest masks + fresh tip markers, risk banner and contact warning.
Every processed frame is appended to live_fast.csv so the per-frame rate is recorded.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.segmentation.predictor import PALETTE_BGR

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
RISK_BGR = {"Minimal": (91, 154, 58), "Low": (74, 191, 156), "Medium": (32, 160, 240), "High": (37, 48, 217),
            "Unavailable": (180, 183, 184)}


class FastLive:
    def __init__(self, warn_s: float = 1.0, min_approach_px_s: float = 20.0, out_csv: Path | None = None,
                 risk_cfg: dict | None = None):
        self.warn_s, self.min_approach = warn_s, min_approach_px_s
        # per-frame risk: same rule-based formula and config as the full pipeline (src/risk/rule_based.py)
        self.rc = {"proximity_ref_frac": 0.08, "gate_mult": 2.0, "speed_ref_frac_per_s": 0.6, "approach_ref_frac_per_s": 0.3,
                   "accel_ref_frac_per_s2": 12.0, "weights": {"proximity": 0.5, "approach": 0.2, "speed": 0.15, "abrupt": 0.15},
                   "smoothing_alpha": 0.4, "thresholds": {"low": 0.25, "medium": 0.5, "high": 0.75}, **(risk_cfg or {})}
        self.diag = None
        self.zones = None                               # suggested Go / No-Go dissection zones (latest keyframe)
        self.score: float | None = None
        self.why = ""
        self.tips: dict[str, np.ndarray] = {}          # name -> (x, y) float32
        self.prev_gray: np.ndarray | None = None
        self.overlay: np.ndarray | None = None          # colour layer of the latest keyframe masks
        self.overlay_mask: np.ndarray | None = None
        self.dist_maps: dict[str, np.ndarray] = {}      # target -> distance map (px)
        self.inside_maps: dict[str, np.ndarray] = {}
        self.state: dict[str, dict] = {}                # name -> {dist, speed, approach, t}
        self.risk = "Unavailable"
        self.t0 = time.perf_counter()
        self.frames = 0
        self.fps = 0.0
        self._last_t = None
        self.out_csv = out_csv
        self._csv = None
        self._w = None
        self.series: list[dict] = []                    # live charts: every frame of the whole session
        self.events: list[dict] = []                    # live event log (warnings, risk changes, bleeding, CVS)
        self.info: dict = {}                            # latest keyframe extras (score, bleeding, CVS, reason)
        self._last_sample = 0.0
        self._warn_t: float | None = None
        self._risk_t: dict[str, float] = {}
        self._nogo_t: dict[str, float] = {}
        self._prev = {"warning": "", "risk": None, "bleeding": False, "cvs": False}
        self.lock = threading.Lock()  # keyframes come from the analysis thread, frames from the capture thread

    # -- keyframe from the full pipeline ---------------------------------------------------------------------------
    def keyframe(self, raw: np.ndarray, row: dict, inst_labels, tissue_labels, tissue_names: list[str],
                 targets: list[str], kp_names: list[str], zones=None) -> None:
        """Full-analysis result. The heavy part (colour layer, distance maps) is built WITHOUT the lock, so the
        capture thread is never blocked; only the final swap is locked."""
        h, w = raw.shape[:2]
        layer = np.zeros_like(raw)
        mask = np.zeros((h, w), bool)
        if tissue_labels is not None and tissue_labels.shape[:2] == (h, w):
            for i, name in enumerate(tissue_names):
                if name in targets:
                    m = tissue_labels == i
                    layer[m] = PALETTE_BGR[i % len(PALETTE_BGR)]
                    mask |= m
        if inst_labels is not None and inst_labels.shape[:2] == (h, w):
            m = inst_labels > 0
            layer[m] = (200, 60, 230)
            mask |= m
        dm, im = {}, {}
        if tissue_labels is not None and tissue_labels.shape[:2] == (h, w):
            for t in targets:
                if t in tissue_names:
                    tm = (tissue_labels == tissue_names.index(t)).astype(np.uint8)
                    if tm.any():
                        dm[t] = cv2.distanceTransform(1 - tm, cv2.DIST_L2, 3)
                        im[t] = tm.astype(bool)
        tips = {}
        for n in kp_names:
            x, y, st = row.get(f"{n}_x"), row.get(f"{n}_y"), row.get(f"{n}_kp_status")
            if x is not None and y is not None and x == x and st in ("ok", "low_confidence"):
                tips[n] = np.array([x, y], np.float32)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        with self.lock:
            self.diag = float(np.hypot(h, w))  # swap in (re-anchors the tracked tips to the keyframe detections)
            self.overlay, self.overlay_mask = layer, mask
            self.zones = zones if zones is not None and zones.available and zones.go.shape == (h, w) else None
            self.dist_maps, self.inside_maps = dm, im
            self.tips = tips
            self.prev_gray = gray
            if self.score is None:  # until the first tracked frame, show the keyframe's risk
                self.risk = row.get("risk_category") or "Unavailable"
            num = lambda v: None if v is None or v != v else float(v)
            self.info = {"risk_score": num(row.get("risk_score")), "bleeding": bool(row.get("bleeding_detected")),
                         "cvs": bool(row.get("cvs_achieved")), "explanation": row.get("risk_explanation") or "",
                         "cvs_c": [num(row.get(f"cvs_c{i}")) for i in (1, 2, 3)],
                         "zones": bool(zones is not None and zones.available),
                         "go_frac": num(row.get("go_zone_frac")), "nogo_frac": num(row.get("nogo_zone_frac"))}

    # -- every camera frame ----------------------------------------------------------------------------------------
    def frame(self, img: np.ndarray) -> dict:
        """Capture thread: track tips and update distance / speed / time to contact for this frame (no drawing)."""
        with self.lock:
            return self._frame(img)

    def _frame(self, img: np.ndarray) -> dict:
        now = time.perf_counter()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.tips and self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            names = list(self.tips)
            p0 = np.array([self.tips[n] for n in names], np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **LK)
            for n, p, ok in zip(names, p1.reshape(-1, 2), st.reshape(-1)):
                if ok:
                    self.tips[n] = p
                else:
                    self.tips.pop(n, None)
        self.prev_gray = gray
        dt = (now - self._last_t) if self._last_t else None
        self._last_t = now
        self.frames += 1
        if dt:
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-3) if self.fps else 1.0 / max(dt, 1e-3)

        best = None
        h, w = img.shape[:2]
        for n, p in self.tips.items():
            x, y = int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1))
            near = None
            for t, dmap in self.dist_maps.items():
                d = 0.0 if self.inside_maps[t][y, x] else float(dmap[y, x])
                if near is None or d < near[1]:
                    near = (t, d)
            s = self.state.get(n)
            if near and s and dt and s.get("target") == near[0]:
                approach = (s["dist"] - near[1]) / dt
                s["approach"] = 0.7 * s.get("approach", approach) + 0.3 * approach
                spd = float(np.hypot(*(p - s["p"]))) / dt
                prev_spd = s.get("speed", spd)
                s["speed"] = 0.7 * prev_spd + 0.3 * spd
                acc = abs(s["speed"] - prev_spd) / dt
                s["accel"] = 0.7 * s.get("accel", acc) + 0.3 * acc
            s = {**(s or {}), "p": p.copy(), "target": near[0] if near else None, "dist": near[1] if near else None,
                 "zone": self.zones.zone_at(x, y) if self.zones is not None else ""}
            self.state[n] = s
            if near:
                a = s.get("approach", 0.0)
                ttc = 0.0 if near[1] <= 0 else (near[1] / a if a > self.min_approach else None)
                if ttc is not None and ttc <= self.warn_s and (best is None or ttc < best[0]):
                    best = (ttc, n, near[0])
        for n in list(self.state):
            if n not in self.tips:
                self.state.pop(n)
        self._frame_risk()

        warning = ""
        if best:
            who = best[1].replace("_tip", " tip").replace("_", " ")
            warning = f"{who} touching {best[2]}" if best[0] <= 0.05 else f"{who} -> {best[2]} in {best[0]:.1f} s"
        self.latest = (img, {n: p.copy() for n, p in self.tips.items()}, warning, self.frames)
        info = {"fps": round(self.fps, 1), "warning": warning, "tips": len(self.tips), "risk": self.risk, "risk_score": self.score}
        self._log(now, best, warning)
        self._sample(now, warning)
        return info

    def _frame_risk(self) -> None:
        """Risk for THIS frame from the tracked tips (rule-based score, EMA smoothed per tip; frame = worst tip)."""
        c, w = self.rc, self.rc["weights"]
        diag = self.diag or 1.0
        clip = lambda v: max(0.0, min(1.0, v))
        best = None
        for n, s in self.state.items():
            d = s.get("dist")
            f_speed = clip((s.get("speed") or 0.0) / diag / c["speed_ref_frac_per_s"])
            f_abrupt = clip((s.get("accel") or 0.0) / diag / c["accel_ref_frac_per_s2"])
            if d is not None:
                dn = d / diag
                f_prox = 1.0 if d <= 0 else clip(1.0 - dn / c["proximity_ref_frac"])
                gate = 1.0 if d <= 0 else clip(1.0 - dn / (c["gate_mult"] * c["proximity_ref_frac"]))
                f_app = clip(max(0.0, s.get("approach") or 0.0) / diag / c["approach_ref_frac_per_s"])
                parts = {"proximity": w["proximity"] * f_prox, "approach": gate * w["approach"] * f_app,
                         "speed": gate * w["speed"] * f_speed, "abrupt": gate * w["abrupt"] * f_abrupt}
            else:  # no target tissue visible: motion only
                den = w["speed"] + w["abrupt"]
                parts = {"speed": w["speed"] * f_speed / den, "abrupt": w["abrupt"] * f_abrupt / den}
            raw = clip(sum(parts.values()))
            a = c["smoothing_alpha"]
            s["risk"] = raw if s.get("risk") is None else a * raw + (1 - a) * s["risk"]
            if best is None or s["risk"] > best[0]:
                best = (s["risk"], n, s, parts)
        if best is None:
            # no instrument tip in view -> nothing can touch tissue: no risk (score 0) for this frame
            self.score, self.risk, self.why = 0.0, "Minimal", "No instrument tip detected - no risk"
            return
        th = c["thresholds"]
        sc = best[0]
        self.score = round(sc, 4)
        self.risk = "High" if sc >= th["high"] else "Medium" if sc >= th["medium"] else "Low" if sc >= th["low"] else "Minimal"
        n, s, parts = best[1], best[2], best[3]
        bits = []
        if s.get("dist") is not None:
            bits.append(f"tip {s['dist']:.0f}px from {s.get('target')} (+{parts['proximity']:.2f})")
        if parts.get("approach", 0) >= 0.03:
            bits.append(f"approaching {s.get('approach', 0):.0f}px/s (+{parts['approach']:.2f})")
        if parts.get("speed", 0) >= 0.03:
            bits.append(f"moving {s.get('speed', 0):.0f}px/s (+{parts['speed']:.2f})")
        if parts.get("abrupt", 0) >= 0.03:
            bits.append(f"abrupt change (+{parts['abrupt']:.2f})")
        self.why = f"{n}: {self.risk} - " + ("; ".join(bits) or "no nearby tissue, slow motion")

    def _event(self, t: float, kind: str, label: str) -> None:
        self.events.append({"t": round(t, 2), "kind": kind, "label": label})

    def _sample(self, now: float, warning: str) -> None:
        """Record a chart sample for EVERY frame and log state changes as events (called under the lock)."""
        t = now - self.t0
        p = self._prev
        if warning:  # one event per warning episode (the warning flickers frame to frame)
            if self._warn_t is None or t - self._warn_t > 2.0:
                self._event(t, "Contact warning", warning)
            self._warn_t = t
        if self.risk != p["risk"] and self.risk in ("Medium", "High"):  # rises only, once per 3 s per level
            if t - self._risk_t.get(self.risk, -9.0) > 3.0:
                self._event(t, "Risk", f"Risk rose to {self.risk}")
            self._risk_t[self.risk] = t
        if self.info.get("bleeding") and not p["bleeding"]:
            self._event(t, "Bleeding", "Bleeding flagged")
        if self.info.get("cvs") and not p["cvs"]:
            self._event(t, "CVS", "Critical View of Safety reached")
        nogo = sorted(n for n, st in self.state.items() if st.get("zone") == "no_go")
        for n in nogo:
            if t - self._nogo_t.get(n, -9.0) > 2.0:
                self._event(t, "No-Go zone", f"{n.replace('_tip', ' tip').replace('_', ' ')} in No-Go zone")
            self._nogo_t[n] = t
        self._prev = {"warning": warning, "risk": self.risk, "bleeding": bool(self.info.get("bleeding")),
                      "cvs": bool(self.info.get("cvs"))}
        self._last_sample = now
        r = lambda v: None if v is None else round(float(v), 1)
        tips = {}
        for n, s in self.state.items():
            d, a = s.get("dist"), s.get("approach", 0.0)
            ttc = None if d is None else (0.0 if d <= 0 else (d / a if a > self.min_approach else None))
            tips[n] = {"dist": r(d), "speed": r(s.get("speed")), "approach": r(a), "ttc": r(ttc), "target": s.get("target"),
                       "zone": s.get("zone", "")}
        self.series.append({"t": round(t, 3), "frame": self.frames, "risk": self.risk, "score": self.score,
                            "warning": warning, "bleeding": bool(self.info.get("bleeding")), "tips": tips})

    def snapshot(self, since: int = 0) -> dict:
        """Live charts / events / readouts: every frame after frame number `since` (series[i] is frame i + 1)."""
        with self.lock:
            return {"series": self.series[max(0, int(since)):], "events": list(self.events),
                    "info": {**self.info, "risk_score": self.score, "explanation": self.why}, "risk": self.risk, "fps": round(self.fps, 1)}

    def render(self):
        """Draw the newest tracked frame (called from the display thread, not the capture thread)."""
        with self.lock:
            latest, overlay, omask, risk, fps = getattr(self, "latest", None), self.overlay, self.overlay_mask, self.risk, self.fps
            zones = self.zones
        if latest is None:
            return None, -1
        img, tips, warning, n = latest
        h, w = img.shape[:2]
        out = img.copy()
        if overlay is not None and overlay.shape == img.shape:
            blend = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            out[omask] = blend[omask]
        if zones is not None and zones.go.shape == img.shape[:2]:
            from src.safety.dissection_zones import draw as draw_zones

            out = draw_zones(out, zones)
        col = RISK_BGR.get(risk, RISK_BGR["Unavailable"])
        cv2.rectangle(out, (0, 0), (w, 24), col, -1)
        cv2.putText(out, f"Risk: {risk}   {fps:4.1f} fps", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        for p in tips.values():
            c = (int(p[0]), int(p[1]))
            cv2.circle(out, c, 9, (37, 48, 217), 1, cv2.LINE_AA)
            cv2.circle(out, c, 5, (37, 48, 217), 1, cv2.LINE_AA)
            cv2.circle(out, c, 2, (37, 48, 217), -1, cv2.LINE_AA)
        if warning:
            from src.pipeline.overlay import draw_contact_warning

            out = draw_contact_warning(out, warning, n)
        return out, n

    def _log(self, now: float, best, warning: str) -> None:
        if self.out_csv is None:
            return
        if self._csv is None:
            self.out_csv.parent.mkdir(parents=True, exist_ok=True)
            self._csv = open(self.out_csv, "w", newline="", encoding="utf-8")
            self._w = csv.writer(self._csv)
            self._w.writerow(["t_s", "tip", "x", "y", "target", "distance_px", "approach_px_s", "speed_px_s", "ttc_s", "warning", "risk_category", "risk_score"])
        t = round(now - self.t0, 3)
        for n, s in self.state.items():
            ttc = ""
            if s.get("dist") is not None:
                a = s.get("approach", 0.0)
                ttc = 0.0 if s["dist"] <= 0 else (round(s["dist"] / a, 2) if a > self.min_approach else "")
            self._w.writerow([t, n, round(float(s["p"][0]), 1), round(float(s["p"][1]), 1), s.get("target"),
                              None if s.get("dist") is None else round(s["dist"], 1),
                              round(s.get("approach", 0.0), 1), round(s.get("speed", 0.0), 1), ttc, warning, self.risk, self.score])

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
            self._csv = None
