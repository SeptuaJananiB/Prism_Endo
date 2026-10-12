"""Keypoint sources used by the pipeline (DLC predictions or mask heuristic)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.tracking.dlc_io import DLCPredictions
from src.tracking.keypoints import Keypoint, KeypointSpec, gate
from src.tracking.auto_tips import components, tip_from_component
from src.tracking.mask_tip import estimate_tip, field_of_view_mask, fov_edge_distance


class MaskTipKeypointSource:
    """Fallback: tip = far end of the predicted instrument-class mask.

    With max_instances > 1 every sufficiently large connected component of the
    class mask is treated as a separate instrument; detections are matched to
    fixed slots (name, name_2, ...) by minimum total distance to each slot's
    previous position (Hungarian assignment, gated by `max_jump_px`), so a slot
    keeps following the same instrument. Instruments whose masks touch/merge
    form one component and cannot be separated.
    """

    name = "mask_heuristic"

    def __init__(self, specs: list[KeypointSpec], instrument_class_names: list[str],
                 min_confidence: float = 0.3, min_area: int = 300, edge_margin: int = 12,
                 max_jump_px: float = 150.0, forget_after: int = 12, fov_every: int = 25,
                 landmarks: dict | None = None):
        self.specs = specs
        # instrument_class -> {"landmark": "far_end"|"hinge", "jaw_px": float}: geodesic landmark (auto_tips) instead
        # of the straight-line far end. Absent class -> original estimate_tip.
        self.landmarks = landmarks or {}
        self.class_index = {n: i for i, n in enumerate(instrument_class_names)}
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.edge_margin = edge_margin
        self.max_jump_px = max_jump_px
        self.forget_after = forget_after
        self.fov_every = max(1, fov_every)
        self._fov = None
        self._fov_age = 0
        self._prev: dict[str, tuple[float, float] | None] = {}
        self._unseen: dict[str, int] = {}
        for s in specs:
            if s.instrument_class not in self.class_index:
                raise ValueError(f"Keypoint '{s.name}' refers to instrument class '{s.instrument_class}', "
                                 f"which the instrument model does not have: {instrument_class_names}")

    def _field_of_view(self, frame_bgr: np.ndarray) -> np.ndarray:
        # The endoscope's circular field of view is static within a video: refresh periodically only.
        if self._fov is None or self._fov.shape != frame_bgr.shape[:2] or self._fov_age >= self.fov_every:
            self._fov, self._fov_age = field_of_view_mask(frame_bgr), 0
            self._dist_edge = fov_edge_distance(self._fov)
        self._fov_age += 1
        return self._fov

    def _keypoint(self, name: str, est: dict | None) -> Keypoint:
        if est is None:
            return Keypoint(name, source=self.name, status="missing")
        kp = Keypoint(name, est["x"], est["y"], est["confidence"], self.name,
                      extra={"entry": est["entry"], "method": est["method"]})
        return gate(kp, self.min_confidence)

    def _estimate(self, mask: np.ndarray, fov: np.ndarray, cls: str) -> dict | None:
        lm = self.landmarks.get(cls)
        if not lm:
            return estimate_tip(mask, fov, self.min_area, self.edge_margin, dist_edge=self._dist_edge)
        comps = components(mask, self.min_area, keep=1)
        if not comps:
            return None
        est = tip_from_component(comps[0], self._dist_edge, lm.get("landmark", "far_end"),
                                 edge_margin=self.edge_margin, jaw_px=lm.get("jaw_px"), step=int(lm.get("step", 2)),
                                 jaw_rel=lm.get("jaw_rel"))
        jaw = lm.get("jaw_px")
        if est is None or (jaw is not None and est["geo_len"] < jaw + 10):
            # Visible instrument shorter than the jaw (e.g. hugging the field-of-view border): the hinge cannot be
            # located. Fall back to the straight-line far end so proximity/risk keep a point; flagged in `method`.
            fb = estimate_tip(mask, fov, self.min_area, self.edge_margin, dist_edge=self._dist_edge)
            if fb is not None:
                fb["method"] = "far_end_fallback(" + fb["method"] + ")"
            return fb
        return est

    def _instances(self, mask: np.ndarray, fov: np.ndarray, n: int, cls: str = "") -> list[dict]:
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        order = sorted(range(1, cnt), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
        dets = []
        for i in order[:n]:
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                break
            est = self._estimate(lab == i, fov, cls)
            if est is not None:
                dets.append(est)
        return dets

    def _assign(self, names: list[str], dets: list[dict]) -> dict[str, dict | None]:
        from scipy.optimize import linear_sum_assignment

        out: dict[str, dict | None] = {n: None for n in names}
        free = list(range(len(dets)))
        tracked = [n for n in names if self._prev.get(n) is not None]
        if tracked and dets:
            cost = np.array([[np.hypot(d["x"] - self._prev[n][0], d["y"] - self._prev[n][1]) for d in dets]
                             for n in tracked])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= self.max_jump_px:
                    out[tracked[r]] = dets[c]
                    free.remove(c)
        # New detections (largest first) fill empty slots, preferring slots with no recent history.
        empty = sorted([n for n in names if out[n] is None], key=lambda n: self._prev.get(n) is not None)
        for c, n in zip(free, empty):
            out[n] = dets[c]
        for n in names:
            if out[n] is not None:
                self._prev[n], self._unseen[n] = (out[n]["x"], out[n]["y"]), 0
            else:
                self._unseen[n] = self._unseen.get(n, 0) + 1
                if self._unseen[n] > self.forget_after:
                    self._prev[n] = None
        return out

    def __call__(self, frame_bgr: np.ndarray, frame_index: int,
                 instrument_labels: np.ndarray | None) -> list[Keypoint]:
        if instrument_labels is None:
            return [Keypoint(n, source=self.name, status="missing") for s in self.specs for n in s.slot_names()]
        fov = self._field_of_view(frame_bgr)
        out = []
        for s in self.specs:
            mask = instrument_labels == self.class_index[s.instrument_class]
            if s.max_instances <= 1:
                out.append(self._keypoint(s.name, self._estimate(mask, fov, s.instrument_class)))
                continue
            assigned = self._assign(s.slot_names(), self._instances(mask, fov, s.max_instances, s.instrument_class))
            out += [self._keypoint(n, assigned[n]) for n in s.slot_names()]
        return out


class DLCKeypointSource:
    """Reads keypoints for each frame from a DeepLabCut prediction file
    (produced by `deeplabcut.analyze_videos` on the same video)."""

    name = "dlc"

    def __init__(self, specs: list[KeypointSpec], predictions_file: str | Path,
                 min_confidence: float = 0.6, individual: str | None = None):
        self.specs = specs
        self.pred = DLCPredictions(predictions_file, individual)
        self.min_confidence = min_confidence
        missing = [s.dlc_bodypart for s in specs if s.dlc_bodypart not in self.pred.bodyparts]
        # Extra instance slots (e.g. grasper_tip_2) are optional: absent bodyparts stay "missing".
        self.slots = [(n, bp) for sp in specs for n, bp in zip(sp.slot_names(), sp.slot_bodyparts())]
        self.absent_slots = [n for n, bp in self.slots if bp not in self.pred.bodyparts]
        if missing:
            raise ValueError(f"DLC file {predictions_file} lacks bodyparts {missing}; "
                             f"it has {self.pred.bodyparts}")

    def __call__(self, frame_bgr, frame_index: int, instrument_labels=None) -> list[Keypoint]:
        out = []
        for name, bp in self.slots:
            r = None if name in self.absent_slots else self.pred.get(frame_index, bp)
            if r is None:
                out.append(Keypoint(name, source=self.name, status="missing"))
                continue
            x, y, lik = r
            out.append(gate(Keypoint(name, x, y, lik, self.name), self.min_confidence))
        return out


def find_dlc_predictions(video: str | Path, folder: str | Path | None = None) -> Path | None:
    """Locate `<video_stem>DLC_*.h5|.csv` next to the video (or in `folder`)."""
    video = Path(video)
    folder = Path(folder) if folder else video.parent
    hits = sorted(folder.glob(f"{video.stem}DLC*.h5")) + sorted(folder.glob(f"{video.stem}DLC*.csv"))
    hits = [h for h in hits if "_meta" not in h.stem and "_full" not in h.stem]
    return hits[0] if hits else None


class HybridKeypointSource:
    """DeepLabCut where it is plausible, the mask-tip estimate elsewhere.

    For each slot whose bodypart is in `dlc_slots`, the DLC point is used when its likelihood is >= `accept_conf`
    OR it lies on the segmentation mask of that slot's instrument class (within `on_mask_px`). This rejects DLC
    points that land on tissue (the main DLC failure mode on unseen videos). Otherwise the mask-tip keypoint is used.
    """

    name = "hybrid"

    def __init__(self, mask_source: "MaskTipKeypointSource", dlc_source: "DLCKeypointSource",
                 dlc_slots: list[str] | None = None, accept_conf: float = 0.6, on_mask_px: int = 12):
        self.mask, self.dlc = mask_source, dlc_source
        self.dlc_slots = set(dlc_slots) if dlc_slots else None
        self.accept_conf, self.on_mask_px = accept_conf, on_mask_px
        self.slot_class = {n: s.instrument_class for s in mask_source.specs for n in s.slot_names()}

    def _on_mask(self, kp: Keypoint, labels: np.ndarray | None) -> bool:
        if labels is None or kp.x is None or not np.isfinite(kp.x):
            return False
        cls = self.mask.class_index.get(self.slot_class.get(kp.name))
        if cls is None:
            return False
        h, w = labels.shape
        x, y, r = int(round(kp.x)), int(round(kp.y)), self.on_mask_px
        win = labels[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        return bool(win.size) and bool((win == cls).any())

    def __call__(self, frame_bgr, frame_index: int, instrument_labels=None) -> list[Keypoint]:
        mk = {k.name: k for k in self.mask(frame_bgr, frame_index, instrument_labels)}
        dk = {k.name: k for k in self.dlc(frame_bgr, frame_index, instrument_labels)}
        out = []
        for name, m in mk.items():
            d = dk.get(name)
            use = (d is not None and (self.dlc_slots is None or name in self.dlc_slots) and d.x is not None
                   and np.isfinite(d.x) and (d.confidence >= self.accept_conf or self._on_mask(d, instrument_labels)))
            if use:
                conf = max(d.confidence, m.confidence if m.status == "ok" else 0.0, self.mask.min_confidence)
                out.append(Keypoint(name, d.x, d.y, conf, self.name, status="ok",
                                    extra={"method": "dlc" if d.confidence >= self.accept_conf else "dlc_on_mask",
                                           "dlc_likelihood": d.confidence}))
            else:
                m.extra = {**(m.extra or {}), "method": "mask_tip_fallback:" + str((m.extra or {}).get("method", ""))}
                m.source = self.name
                out.append(m)
        return out
