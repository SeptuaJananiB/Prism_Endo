"""TipNet: a heatmap keypoint network for laparoscopic instrument tips, trained by backpropagation.

U-Net (ResNet-18 encoder, ImageNet init) -> 2 heatmaps at input resolution:
  channel 0 = L-hook tip (single peak), channel 1 = grasper jaw hinge (up to 2 peaks, identity-free).
Targets are Gaussian blobs around PSEUDO labels (scripts/auto_label_all.py); loss = CenterNet penalty-reduced focal
loss. Inference decodes peaks with non-maximum suppression and a 3x3 soft-argmax refinement.
"""

from __future__ import annotations

import json
import random

import cv2
import numpy as np

IN_H, IN_W = 256, 448            # network input (same as the segmentation models)
ORIG_H, ORIG_W = 480, 854        # CholecSeg8k frame size
CHANNELS = ("lhook_tip", "grasper_hinge")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(pretrained: bool = True):
    import segmentation_models_pytorch as smp

    m = smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None, classes=len(CHANNELS))
    m.segmentation_head[0].bias.data.fill_(-2.19)  # CenterNet prior (p = 0.1): stable start for the focal loss
    return m


def preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    return ((img - MEAN) / STD).transpose(2, 0, 1).copy()


def gaussian_heatmaps(points: list[list], h: int = IN_H, w: int = IN_W, sigma: float = 3.0) -> np.ndarray:
    """points[c] = list of (x, y) in INPUT pixels for channel c -> (C, h, w) float32, peak pixel exactly 1."""
    hm = np.zeros((len(CHANNELS), h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for c, pts in enumerate(points):
        for x, y in pts:
            if not (0 <= x < w and 0 <= y < h):
                continue
            g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
            hm[c] = np.maximum(hm[c], g)
            hm[c, int(round(min(y, h - 1))), int(round(min(x, w - 1)))] = 1.0
    return hm


def focal_loss(logits, target, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced pixel-wise focal loss (written with basic ops so it runs on DirectML)."""
    import torch

    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def decode(heat: np.ndarray, thr: float = 0.3, max_grasper: int = 2, nms: int = 15) -> dict:
    """heat: (C, h, w) sigmoid heatmaps at INPUT resolution -> points in ORIGINAL pixels with scores."""
    sx, sy = ORIG_W / heat.shape[2], ORIG_H / heat.shape[1]
    out = {}
    for c, name in enumerate(CHANNELS):
        hm = heat[c]
        dil = cv2.dilate(hm, np.ones((nms, nms), np.uint8))
        peaks = np.argwhere((hm >= dil) & (hm >= thr))
        peaks = sorted(peaks, key=lambda p: -hm[p[0], p[1]])[: (1 if c == 0 else max_grasper)]
        pts = []
        for y, x in peaks:
            y0, y1, x0, x1 = max(0, y - 1), min(hm.shape[0], y + 2), max(0, x - 1), min(hm.shape[1], x + 2)
            win = hm[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cx, cy = float((xx * win).sum() / win.sum()), float((yy * win).sum() / win.sum())
            pts.append(((cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5, float(hm[y, x])))
        out[name] = pts
    return out


def labels_from_row(row) -> list[list]:
    """all_frames.csv row -> per-channel point lists in ORIGINAL pixels."""
    def _j(v):  # pandas may read the JSON text "null" as NaN
        return json.loads(v) if isinstance(v, str) and v else None

    lh, gr = _j(row["lhook"]), _j(row["grasper"]) or []
    return [[lh[:2]] if lh else [], [g[:2] for g in gr]]


def augment(img: np.ndarray, pts: list[list], rng: random.Random):
    """Geometric (flip, scale/rotate/shift) + photometric augmentation on an ORIGINAL-size frame and its points."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        pts = [[[w - 1 - x, y] for x, y in p] for p in pts]
    if rng.random() < 0.7:
        a, s = rng.uniform(-15, 15), rng.uniform(0.85, 1.2)
        tx, ty = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
        m[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT)
        pts = [[[m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]] for x, y in p] for p in pts]
    if rng.random() < 0.8:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    return img, pts
