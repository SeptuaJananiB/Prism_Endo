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
    def __init__(self, warn_s: float = 1.0, min_approach_px_s: float = 20.0, out_csv: Path | None = None):
        self.warn_s, self.min_approach = warn_s, min_approach_px_s
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
        self.lock = threading.Lock()  # keyframes come from the analysis thread, frames from the capture thread

    # -- keyframe from the full pipeline ---------------------------------------------------------------------------
    def keyframe(self, raw: np.ndarray, row: dict, inst_labels, tissue_labels, tissue_names: list[str],
                 targets: list[str], kp_names: list[str]) -> None:
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
        with self.lock:  # swap in (re-anchors the tracked tips to the keyframe detections)
            self.overlay, self.overlay_mask = layer, mask
            self.dist_maps, self.inside_maps = dm, im
            self.tips = tips
            self.prev_gray = gray
            self.risk = row.get("risk_category") or "Unavailable"

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
                s["speed"] = 0.7 * s.get("speed", spd) + 0.3 * spd
            s = {**(s or {}), "p": p.copy(), "target": near[0] if near else None, "dist": near[1] if near else None}
            self.state[n] = s
            if near:
                a = s.get("approach", 0.0)
                ttc = 0.0 if near[1] <= 0 else (near[1] / a if a > self.min_approach else None)
                if ttc is not None and ttc <= self.warn_s and (best is None or ttc < best[0]):
                    best = (ttc, n, near[0])
        for n in list(self.state):
            if n not in self.tips:
                self.state.pop(n)

        warning = ""
        if best:
            who = best[1].replace("_tip", " tip").replace("_", " ")
            warning = f"{who} touching {best[2]}" if best[0] <= 0.05 else f"{who} -> {best[2]} in {best[0]:.1f} s"
        self.latest = (img, {n: p.copy() for n, p in self.tips.items()}, warning, self.frames)
        info = {"fps": round(self.fps, 1), "warning": warning, "tips": len(self.tips)}
        self._log(now, best, warning)
        return info

    def render(self):
        """Draw the newest tracked frame (called from the display thread, not the capture thread)."""
        with self.lock:
            latest, overlay, omask, risk, fps = getattr(self, "latest", None), self.overlay, self.overlay_mask, self.risk, self.fps
        if latest is None:
            return None, -1
        img, tips, warning, n = latest
        h, w = img.shape[:2]
        out = img.copy()
        if overlay is not None and overlay.shape == img.shape:
            blend = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            out[omask] = blend[omask]
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
            self._w.writerow(["t_s", "tip", "x", "y", "target", "distance_px", "approach_px_s", "speed_px_s", "ttc_s", "warning"])
        t = round(now - self.t0, 3)
        for n, s in self.state.items():
            ttc = ""
            if s.get("dist") is not None:
                a = s.get("approach", 0.0)
                ttc = 0.0 if s["dist"] <= 0 else (round(s["dist"] / a, 2) if a > self.min_approach else "")
            self._w.writerow([t, n, round(float(s["p"][0]), 1), round(float(s["p"][1]), 1), s.get("target"),
                              None if s.get("dist") is None else round(s["dist"], 1),
                              round(s.get("approach", 0.0), 1), round(s.get("speed", 0.0), 1), ttc, warning])

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
            self._csv = None
