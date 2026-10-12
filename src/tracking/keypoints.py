"""Keypoint data structures and the keypoint-source interface.

Segmentation (pixel masks) and keypoint tracking are separate modules. The
pipeline asks a KeypointSource for the configured keypoints of each frame;
sources are:

  * DLCKeypointSource      - predictions from a trained DeepLabCut model
                             (src/tracking/dlc_io.py). This is the intended
                             primary source.
  * MaskTipKeypointSource  - a geometric heuristic that places the tip at the
                             far end of a predicted instrument mask
                             (src/tracking/mask_tip.py). It is a fallback so
                             the pipeline can run before DLC is trained; its
                             "confidence" is a heuristic quality score, not
                             a calibrated probability, and its points are
                             never ground truth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class KeypointSpec:
    name: str                       # e.g. "lhook_tip"
    instrument_class: str | None    # segmentation class the point belongs to, e.g. "l_hook"
    dlc_bodypart: str | None = None  # DLC bodypart name (defaults to `name`)
    max_instances: int = 1           # >1: track several instruments of this class (slots name, name_2, ...)

    def slot_names(self) -> list[str]:
        return [self.name] + [f"{self.name}_{k}" for k in range(2, self.max_instances + 1)]

    def slot_bodyparts(self) -> list[str]:
        bp = self.dlc_bodypart or self.name
        return [bp] + [f"{bp}_{k}" for k in range(2, self.max_instances + 1)]


@dataclass
class Keypoint:
    name: str
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    source: str = "none"            # "dlc" | "mask_heuristic" | "none"
    status: str = "missing"         # "ok" | "low_confidence" | "missing"
    extra: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "ok" and self.x is not None and self.y is not None \
            and math.isfinite(self.x) and math.isfinite(self.y)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return d


def gate(kp: Keypoint, min_confidence: float) -> Keypoint:
    """Mark a keypoint low-confidence (but keep its raw coordinates for
    display/export) when below threshold. Low-confidence points are not used
    for motion or proximity."""
    if kp.x is None or kp.y is None or not (math.isfinite(kp.x) and math.isfinite(kp.y)):
        kp.status = "missing"
    elif kp.confidence < min_confidence:
        kp.status = "low_confidence"
    else:
        kp.status = "ok"
    return kp


def specs_from_config(cfg_list: list[dict]) -> list[KeypointSpec]:
    return [KeypointSpec(name=d["name"], instrument_class=d.get("instrument_class"),
                         dlc_bodypart=d.get("dlc_bodypart") or d["name"],
                         max_instances=max(1, int(d.get("max_instances", 1)))) for d in cfg_list]


def slot_names(specs: list[KeypointSpec]) -> list[str]:
    """All per-instance keypoint names, in a fixed order (stable CSV columns)."""
    return [n for s in specs for n in s.slot_names()]


def slot_classes(specs: list[KeypointSpec]) -> dict[str, str | None]:
    return {n: s.instrument_class for s in specs for n in s.slot_names()}
