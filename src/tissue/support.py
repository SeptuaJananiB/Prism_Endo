"""Per-class support of a tissue model: how much of each class the model saw
in training and whether it was evaluated on held-out data.

This exists so the pipeline never presents proximity to a structure as
meaningful when the model cannot know that structure:

  untrained    - zero (or almost zero) training frames contain the class; the
                 model cannot detect it, any prediction is noise.
  unevaluated  - present in training but absent from the held-out test split,
                 so detection quality is unknown.
  evaluated    - present in training and in the test split; held-out Dice is
                 reported (from the evaluate_segmentation.py metrics.json).

Support is counted from the dense ground-truth label maps of the splits
stored in the checkpoint metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.segmentation_data import build_lut, discover_samples, read_label, split_samples

SUPPORT_FILE = "class_support.json"


def compute_class_support(ckpt_meta: dict, data_cfg: dict, min_pixels: int = 50,
                          every: int = 1) -> dict:
    """Count, per split, the frames and pixels of every class in the checkpoint."""
    from src.utils.config import resolve_path

    classes = ckpt_meta["classes"]
    names = [c["name"] for c in classes]
    lut = build_lut(classes, data_cfg.get("ignore_values", [255]), data_cfg.get("unlisted", "background"))
    roots = data_cfg["root"] if isinstance(data_cfg["root"], list) else [data_cfg["root"]]
    samples = []
    for r in roots:
        samples += discover_samples(resolve_path(r), data_cfg.get("layout", "cholecseg8k"),
                                    data_cfg.get("label_suffix", "_endo_watershed_mask.png"),
                                    group_by=data_cfg.get("group_by", "clip"))
    splits = split_samples(samples, ckpt_meta.get("splits") or data_cfg["splits"])
    out = {"min_pixels_per_frame": min_pixels, "sampled_every": every, "splits": {}}
    for split, items in splits.items():
        frames = np.zeros(len(names), int)
        pixels = np.zeros(len(names), np.int64)
        for s in items[::every]:
            counts = np.bincount(read_label(s.label, lut).ravel(), minlength=256)[:len(names)]
            pixels += counts
            frames += counts >= min_pixels
        out["splits"][split] = {"frames_evaluated": len(items[::every]),
                                "frames_with_class": dict(zip(names, frames.tolist())),
                                "pixels": dict(zip(names, pixels.tolist()))}
    return out


def reliability(support: dict | None, metrics: dict | None, cls: str,
                min_train_frames: int = 20, min_test_frames: int = 20, fail_dice: float = 0.5) -> dict:
    """Classify one class as evaluated / unevaluated / untrained / unknown."""
    if not support:
        return {"class": cls, "reliability": "unknown",
                "note": f"no {SUPPORT_FILE} next to the checkpoint (run scripts/class_support.py)"}
    tr = support["splits"].get("train", {}).get("frames_with_class", {}).get(cls, 0)
    te = support["splits"].get("test", {}).get("frames_with_class", {}).get(cls, 0)
    dice = None
    if metrics:
        dice = (metrics.get("per_class", {}).get(cls) or {}).get("dice")
    info = {"class": cls, "train_frames": tr, "test_frames": te, "test_dice": dice}
    if tr < min_train_frames:
        info.update(reliability="untrained",
                    note=f"only {tr} training frames contain '{cls}': the model cannot detect it reliably")
    elif te == 0:
        info.update(reliability="unevaluated",
                    note=f"'{cls}' is absent from the held-out test split: detection quality unknown")
    elif te < min_test_frames:
        info.update(reliability="insufficient_test_data",
                    note=f"only {te} held-out test frames contain '{cls}'" +
                         (f" (Dice {dice:.3f} there)" if dice is not None else "") + ": quality not established")
    elif dice is not None and dice < fail_dice:
        info.update(reliability="failed_on_heldout",
                    note=f"held-out test Dice {dice:.3f} < {fail_dice}: do not rely on '{cls}'")
    else:
        info.update(reliability="evaluated",
                    note=f"held-out test Dice {dice:.3f}" if dice is not None else "test metrics not found")
    return info


def load_support(checkpoint: str | Path) -> tuple[dict | None, dict | None]:
    """Read class_support.json and eval_test/metrics.json next to a checkpoint, if present."""
    folder = Path(checkpoint).parent
    sup = folder / SUPPORT_FILE
    met = folder / "eval_test" / "metrics.json"
    support = json.loads(sup.read_text(encoding="utf-8")) if sup.is_file() else None
    metrics = json.loads(met.read_text(encoding="utf-8")) if met.is_file() else None
    return support, metrics
