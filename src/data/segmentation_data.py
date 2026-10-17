"""Segmentation data discovery, label mapping, group-wise splitting and a
PyTorch Dataset.

Supported layouts
-----------------
``cholecseg8k``
    root/<videoXX>/<videoXX_NNNNN>/frame_<n>_endo.png  + frame_<n>_endo_watershed_mask.png
    The watershed mask stores CholecSeg8k class ids (e.g. 31 = grasper,
    32 = L-hook electrocautery, 21 = liver) in every channel; 255 marks the
    thin annotation boundary lines and is ignored. NOTE: `*_endo_mask.png`
    is a *sparse* annotation-tool mask (most pixels 0) and must not be used
    as a dense label map.
``indexed_png``
    root/images/<group>_<anything>.png + root/masks/<same name>.png, where
    the mask stores raw label values that the task config maps to classes.
    <group> (text before the first underscore) is the video/patient id used
    for splitting.

``data.root`` may be one folder or a list of folders (e.g. ``[video01,
data/cholecseg8k/video09]``); samples are discovered across all of them.

Splits are always by *group* (clip / video / patient). Adjacent frames of
one clip are never split between train and test. ``data.group_by`` selects
the group: ``clip`` (default, one CholecSeg8k clip folder of 80 frames) or
``video`` (one Cholec80 video = one procedure). Every Sample also carries
its ``video`` id so leakage across videos can always be checked. Each
Cholec80 video is a separate procedure; patient identity is not released,
so video-level grouping is the closest available proxy for patients.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.device import get_logger

log = get_logger("prism.data")

IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GROUP_BY = ("clip", "video")


def video_id_from_name(name: str) -> str:
    """CholecSeg8k clip folders are named <video>_<first frame>, e.g.
    video01_00080 -> video01; indexed_png stems <video>_<x> likewise."""
    return str(name).split("_")[0]


@dataclass
class Sample:
    image: Path
    label: Path
    group: str
    video: str = ""   # Cholec80 video (procedure) id; derived from `group` if not given

    def __post_init__(self):
        if not self.video:
            self.video = video_id_from_name(self.group)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_lut(classes: list[dict], ignore_values: list[int] | None = None,
              unlisted: str = "background") -> np.ndarray:
    """Build a 256-entry lookup table raw_value -> class index.

    `classes` is an ordered list of ``{name, values}``; index = position.
    Raw values not listed map to class 0 ("background") when
    ``unlisted == "background"``, else to IGNORE_INDEX.
    """
    default = 0 if unlisted == "background" else IGNORE_INDEX
    lut = np.full(256, default, dtype=np.uint8)
    for idx, cls in enumerate(classes):
        for v in cls.get("values", []) or []:
            lut[int(v)] = idx
    for v in ignore_values or []:
        lut[int(v)] = IGNORE_INDEX
    return lut


def class_names(classes: list[dict]) -> list[str]:
    return [c["name"] for c in classes]


def read_label(path: Path, lut: np.ndarray) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Could not read label: {path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    return lut[raw]


# ---------------------------------------------------------------------------
# Discovery and splitting
# ---------------------------------------------------------------------------

def _video_of_clip(clip_dir: Path, root: Path) -> str:
    """Video id of a CholecSeg8k clip folder.

    * root/<video>/<clip>/frames (dataset root given): the clip's parent folder.
    * root/<clip>/frames (a video folder given, e.g. ``root: video01``, or a
      flat folder of clips): the text before the first "_" of the clip name
      (CholecSeg8k names clips <video>_<first frame>), or the root folder's
      name if the clip name has no "_".
    * frames directly in root: the root folder's name.
    """
    parts = clip_dir.relative_to(root).parts
    if len(parts) >= 2:
        return parts[-2]
    if len(parts) == 1:
        return video_id_from_name(parts[0]) if "_" in parts[0] else root.name
    return root.name


def _as_roots(root) -> list[Path]:
    roots = list(root) if isinstance(root, (list, tuple)) else [root]
    if not roots:
        raise ValueError("data.root is an empty list")
    return [Path(r) for r in roots]


def discover_samples(root: str | Path | list, layout: str = "cholecseg8k",
                     label_suffix: str = "_endo_watershed_mask.png",
                     group_by: str = "clip") -> list[Sample]:
    """Find image/label pairs under one root or a list of roots.

    ``group_by="clip"`` (default) groups by clip folder name, exactly as
    before; ``"video"`` groups by video id. Roots are scanned in the given
    order. Files reached through two overlapping roots are kept once; the
    same clip (or indexed_png file name) found in two *different* folders is
    an error, because duplicated data would silently leak across splits.
    """
    if group_by not in GROUP_BY:
        raise ValueError(f"data.group_by must be one of {GROUP_BY}, got '{group_by}'")
    roots = _as_roots(root)
    samples: list[Sample] = []
    seen_files: set[Path] = set()
    origin: dict[str, Path] = {}   # clip name (or indexed_png file name) -> folder it came from
    skipped = 0
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        if layout == "cholecseg8k":
            pairs = []
            for img in sorted(root.rglob("*_endo.png")):
                lab = img.with_name(img.name.replace("_endo.png", label_suffix))
                if lab.exists():
                    clip = img.parent.name
                    pairs.append((img, lab, clip, _video_of_clip(img.parent, root), clip, img.parent))
        elif layout == "indexed_png":
            # No clip level: group = video = text before the first "_" (group_by has no effect).
            img_dir, mask_dir = root / "images", root / "masks"
            pairs = []
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
                lab = mask_dir / (img.stem + ".png")
                if lab.exists():
                    vid = video_id_from_name(img.stem)
                    pairs.append((img, lab, vid, vid, img.name, img_dir))
        else:
            raise ValueError(f"Unknown dataset layout: {layout}")
        for img, lab, clip, vid, key, folder in pairs:
            resolved = img.resolve()
            if resolved in seen_files:
                skipped += 1
                continue
            seen_files.add(resolved)
            prev = origin.setdefault(key, folder.resolve())
            if prev != folder.resolve():
                raise ValueError(f"'{key}' exists in two folders: {prev} and {folder.resolve()}. "
                                 "Remove the duplicate copy or list only one of them in data.root "
                                 "(duplicated data would leak across splits).")
            samples.append(Sample(img, lab, vid if group_by == "video" else clip, vid))
    if skipped:
        log.warning("data.root entries overlap: %d files were reachable from two roots and were "
                    "counted once.", skipped)
    if not samples:
        where = roots[0] if len(roots) == 1 else [str(r) for r in roots]
        raise FileNotFoundError(f"No image/label pairs found under {where} (layout={layout})")
    return samples


def discover_from_config(data_cfg: dict) -> list[Sample]:
    """discover_samples() driven by a config's ``data`` section; relative
    roots are resolved against the project root."""
    from src.utils.config import resolve_path

    roots = [resolve_path(r) for r in _as_roots(data_cfg["root"])]
    return discover_samples(roots if isinstance(data_cfg["root"], (list, tuple)) else roots[0],
                            data_cfg.get("layout", "cholecseg8k"),
                            data_cfg.get("label_suffix", "_endo_watershed_mask.png"),
                            data_cfg.get("group_by", "clip"))


def videos_in_several_splits(split: dict[str, list[Sample]]) -> dict[str, list[str]]:
    """video id -> split names, for every video that occurs in more than one split."""
    where: dict[str, list[str]] = {}
    for name, subset in split.items():
        for v in sorted({s.video for s in subset}):
            where.setdefault(v, []).append(name)
    return {v: names for v, names in where.items() if len(names) > 1}


def split_samples(samples: list[Sample], splits: dict) -> dict[str, list[Sample]]:
    """Assign samples to splits by group id.

    `splits` maps split name -> list of group ids, or the string "rest"
    (all groups not listed elsewhere). Raises if a group appears in more
    than one split, which would leak data. If whole videos are the groups
    this makes a video in two splits an error; with clip groups a video
    shared by several splits is allowed (one-video datasets have no other
    option) but logged as a within-patient split.
    """
    groups = sorted({s.group for s in samples})
    explicit: dict[str, set] = {}
    rest_name = None
    for name, ids in splits.items():
        if ids == "rest":
            rest_name = name
            continue
        explicit[name] = {str(i) for i in (ids or [])}
    seen: dict[str, str] = {}
    for name, ids in explicit.items():
        for g in ids:
            if g in seen:
                raise ValueError(f"Group '{g}' is in both '{seen[g]}' and '{name}' splits (data leakage)")
            seen[g] = name
            if g not in groups:
                raise ValueError(f"Split '{name}' lists unknown group '{g}'. Known: {groups}")
    if rest_name:
        explicit[rest_name] = {g for g in groups if g not in seen}
    out = {name: [s for s in samples if s.group in ids] for name, ids in explicit.items()}
    shared = videos_in_several_splits(out)
    if shared:
        log.warning("Within-patient split: video(s) %s appear in several splits (%s). Splits are by "
                    "clip (temporal block), so the scores measure generalisation across time within "
                    "the same procedure/patient, NOT to new patients. Use data.group_by: video with "
                    ">= 3 videos for a patient-level (video-level) split.",
                    sorted(shared), "; ".join(f"{v}: {', '.join(n)}" for v, n in sorted(shared.items())))
    return out


# ---------------------------------------------------------------------------
# Image preprocessing / augmentation
# ---------------------------------------------------------------------------

def normalize_image(bgr: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 3xHxW, RGB, mean/std normalised."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def augment(image: np.ndarray, label: np.ndarray, rng: random.Random, cfg: dict):
    """Geometric + photometric augmentation applied identically to image and
    label (adapted from Dataset_Utilities/dataset.py). Label borders created
    by rotation are filled with IGNORE_INDEX, not background."""
    h, w = image.shape[:2]
    if rng.random() < cfg.get("hflip_p", 0.5):
        image, label = cv2.flip(image, 1), cv2.flip(label, 1)
    if rng.random() < cfg.get("vflip_p", 0.0):
        image, label = cv2.flip(image, 0), cv2.flip(label, 0)
    rot = cfg.get("rotate_deg", 10)
    lo, hi = cfg.get("scale_range", [0.9, 1.1])
    if rot or lo != 1 or hi != 1:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-rot, rot), rng.uniform(lo, hi))
        image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        label = cv2.warpAffine(label, m, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE_INDEX)
    b_lo, b_hi = cfg.get("brightness", [0.8, 1.2])
    c_lo, c_hi = cfg.get("contrast", [0.8, 1.2])
    out = image.astype(np.float32)
    out = (out - 127.5) * rng.uniform(c_lo, c_hi) + 127.5 * rng.uniform(b_lo, b_hi)
    if rng.random() < cfg.get("specular_p", 0.2):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r = rng.randint(max(2, min(h, w) // 25), max(3, min(h, w) // 8))
        spot = np.zeros((h, w), np.float32)
        cv2.circle(spot, (cx, cy), r, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), sigmaX=r / 2)
        out += spot[..., None] * 120.0
    return np.clip(out, 0, 255).astype(np.uint8), label


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _torch_dataset_base():
    try:
        from torch.utils.data import Dataset
        return Dataset
    except ImportError:  # pragma: no cover
        return object


class SegmentationDataset(_torch_dataset_base()):
    """Yields (image_tensor[3,H,W] float32, label_tensor[H,W] int64)."""

    def __init__(self, samples: list[Sample], lut: np.ndarray, input_size: tuple[int, int],
                 augment_cfg: dict | None = None, cache: bool = True, seed: int = 0):
        import torch  # noqa: F401  (fail early with a clear error if missing)

        self.samples = samples
        self.lut = lut
        self.h, self.w = int(input_size[0]), int(input_size[1])
        self.augment_cfg = augment_cfg
        self.rng = random.Random(seed)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cache = cache

    def __len__(self) -> int:
        return len(self.samples)

    def load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        s = self.samples[idx]
        img = cv2.imread(str(s.image), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image: {s.image}")
        lab = read_label(s.label, self.lut)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self.cache:
            self._cache[idx] = (img, lab)
        return img, lab

    def __getitem__(self, idx: int):
        import torch

        img, lab = self.load(idx)
        if self.augment_cfg:
            img, lab = augment(img, lab, self.rng, self.augment_cfg)
        return torch.from_numpy(normalize_image(img)), torch.from_numpy(lab.astype(np.int64))
