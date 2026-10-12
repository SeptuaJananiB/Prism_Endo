"""Select frames for manual instrument-tip annotation and lay them out for DeepLabCut.

Selection uses the ground-truth instrument SEGMENTATION masks only to know
which frames show an L-hook and/or grasper (and how many graspers); it never
produces tip coordinates. Within each clip, frames are chosen by farthest-point
sampling on small colour thumbnails + instrument-presence features, so the set
is visually diverse and spread over time. Training frames come only from the
TRAIN split; the held-out evaluation set only from the TEST split.

Frames are written as DeepLabCut expects (labeled-data/<video_stem>/img<NN>.png,
NN zero-padded to ceil(log10(n_frames)) digits — checked against
deeplabcut 3.0.2 frame_extraction.py) and decoded from the project's own copy of
the video so pixels match what DLC trains and predicts on.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GRASPER_ID, LHOOK_ID = 31, 32
FRAME_RE = re.compile(r"frame_(\d+)_endo\.png$")


@dataclass
class FrameInfo:
    clip: str
    index: int            # 0-based position within the clip/video
    frame_number: int     # CholecSeg8k frame number
    image: Path
    grasper_px: int
    lhook_px: int
    grasper_instances: int
    feature: np.ndarray

    @property
    def instruments(self) -> str:
        parts = []
        if self.lhook_px:
            parts.append("l_hook")
        if self.grasper_px:
            parts.append("grasper" if self.grasper_instances < 2 else "2x_grasper")
        return "+".join(parts) or "none"


def scan_clip(folder: Path, min_px: int = 300) -> list[FrameInfo]:
    """Instrument presence per frame of one CholecSeg8k clip folder."""
    imgs = sorted((p for p in folder.glob("*_endo.png") if FRAME_RE.search(p.name)),
                  key=lambda p: int(FRAME_RE.search(p.name).group(1)))
    out = []
    for i, img in enumerate(imgs):
        lab = cv2.imread(str(img).replace("_endo.png", "_endo_watershed_mask.png"), cv2.IMREAD_GRAYSCALE)
        g, h = lab == GRASPER_ID, lab == LHOOK_ID
        n, _, stats, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), 8)
        inst = int((stats[1:, cv2.CC_STAT_AREA] >= min_px).sum()) if n > 1 else 0
        thumb = cv2.resize(cv2.imread(str(img)), (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        gp, hp = int(g.sum()), int(h.sum())
        feat = np.concatenate([thumb.ravel(), [2.0 * (gp >= min_px), 2.0 * (hp >= min_px), 1.0 * (inst >= 2)]])
        out.append(FrameInfo(folder.name, i, int(FRAME_RE.search(img.name).group(1)), img,
                             gp if gp >= min_px else 0, hp if hp >= min_px else 0, inst, feat))
    return out


def farthest_point_sample(items: list[FrameInfo], k: int, min_gap: int = 2) -> list[FrameInfo]:
    """Deterministic diverse subset; seeds with the frame showing most instrument pixels."""
    if k <= 0 or not items:
        return []
    feats = np.stack([f.feature for f in items])
    chosen = [int(np.argmax([f.grasper_px + f.lhook_px for f in items]))]
    dist = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < min(k, len(items)):
        d = dist.copy()
        for c in chosen:  # keep a small temporal gap between picks
            lo, hi = max(0, c - min_gap + 1), c + min_gap
            d[lo:hi] = -1
        nxt = int(np.argmax(d))
        if d[nxt] < 0:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
    return [items[i] for i in sorted(chosen)]


def select_frames(clips: dict[str, list[FrameInfo]], total: int) -> list[FrameInfo]:
    """Budget proportional to the number of frames that show an instrument (at least 3 per clip)."""
    cands = {c: [f for f in fr if f.grasper_px or f.lhook_px] for c, fr in clips.items()}
    weight = {c: len(v) for c, v in cands.items() if v}
    if not weight:
        return []
    s = sum(weight.values())
    budget = {c: max(3, int(round(total * w / s))) for c, w in weight.items()}
    while sum(budget.values()) > total:  # trim the largest budgets
        budget[max(budget, key=budget.get)] -= 1
    picked = []
    for c in sorted(budget):
        picked += farthest_point_sample(cands[c], min(budget[c], len(cands[c])))
    return picked


def composition(frames: list[FrameInfo]) -> dict:
    return {"frames": len(frames),
            "with_l_hook": sum(1 for f in frames if f.lhook_px),
            "with_grasper": sum(1 for f in frames if f.grasper_px),
            "with_two_graspers": sum(1 for f in frames if f.grasper_instances >= 2),
            "per_clip": {c: sum(1 for f in frames if f.clip == c) for c in sorted({f.clip for f in frames})}}


def dlc_image_name(index: int, n_frames: int) -> str:
    width = max(1, int(math.ceil(math.log10(n_frames))))
    return f"img{str(index).zfill(width)}.png"


def write_labeled_data(project_dir: Path, video: Path, frames: list[FrameInfo]) -> list[Path]:
    """Decode the selected frames from `video` into labeled-data/<stem>/ with DLC naming."""
    out_dir = project_dir / "labeled-data" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = []
    for f in sorted(frames, key=lambda f: f.index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f.index)
        ok, img = cap.read()
        if not ok:
            raise IOError(f"Could not decode frame {f.index} of {video}")
        path = out_dir / dlc_image_name(f.index, n)
        cv2.imwrite(str(path), img)
        written.append(path)
    cap.release()
    return written


def write_manifest(path: Path, frames: list[FrameInfo], images: dict[tuple[str, int], Path]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "video_frame_index", "cholecseg8k_frame", "image", "instruments_in_gt_mask",
                    "grasper_instances"])
        for f in frames:
            w.writerow([f.clip, f.index, f.frame_number, images[(f.clip, f.index)].name, f.instruments,
                        f.grasper_instances])
