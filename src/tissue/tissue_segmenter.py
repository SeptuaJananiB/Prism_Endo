"""Tissue / anatomy segmentation module (Stage D).

A thin wrapper around a trained multi-class checkpoint (see
configs/tissue_seg.yaml). When no checkpoint exists the module reports
itself unavailable and the pipeline runs in instrument-only mode: no tissue
masks are invented and proximity is reported as unavailable.

Which classes it can recognise is decided by the checkpoint's class list
(i.e. by the annotations it was trained on), never hard-coded here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.segmentation.predictor import SegmentationPredictor
from src.tissue.support import load_support, reliability


class TissueSegmenter:
    def __init__(self, checkpoint: str | Path | None, device: str = "cpu",
                 target_classes: list[str] | None = None, ignore_classes: list[str] | None = None,
                 allow_untrained_targets: bool = False):
        self.available = False
        self.target_info: dict[str, dict] = {}
        self.warnings: list[str] = []
        self.message = ""
        self.predictor = None
        self.class_names: list[str] = []
        self.target_classes = list(target_classes or [])
        self.ignore_classes = set(ignore_classes or [])
        if not checkpoint:
            self.message = "No tissue checkpoint configured: tissue segmentation and proximity disabled."
            return
        path = Path(checkpoint)
        if not path.is_file():
            self.message = (f"Tissue checkpoint not found at {path}: tissue segmentation and proximity "
                            f"disabled (train with: python train_segmentation.py --config configs/tissue_seg.yaml).")
            return
        self.predictor = SegmentationPredictor.from_checkpoint(path, device)
        self.class_names = self.predictor.class_names
        unknown = [c for c in self.target_classes if c not in self.class_names]
        if unknown:
            raise ValueError(f"Proximity targets {unknown} are not classes of the tissue model "
                             f"{self.class_names}. Fix proximity.target_classes in the config.")
        # Per-target reliability from class_support.json / eval_test/metrics.json next to the checkpoint.
        support, metrics = load_support(path)
        for t in self.target_classes:
            info = reliability(support, metrics, t)
            self.target_info[t] = info
            if info["reliability"] == "untrained" and not allow_untrained_targets:
                raise ValueError(f"Proximity target '{t}' cannot be used: {info['note']}. Remove it from "
                                 f"proximity.target_classes (or set proximity.allow_untrained_targets=true "
                                 f"to override for experiments).")
            if info["reliability"] != "evaluated":
                self.warnings.append(f"Proximity target '{t}' is {info['reliability']}: {info['note']}.")
        self.available = True
        self.message = f"Tissue model loaded ({len(self.class_names)} classes)."

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if not self.available:
            return None
        return self.predictor.predict(frame_bgr)

    def class_id(self, name: str) -> int:
        return self.class_names.index(name)
