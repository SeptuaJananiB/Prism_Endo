"""Bleeding detection from a trained segmentation model (SurgBlood public test release, split by clip).

Model: U-Net ResNet-18, 288x512 input (scripts/train_bleeding.py, trained on a Colab T4). Held-out clips: detection
F1 0.937, AUROC 0.942, Dice 0.671 on bleeding frames (outputs/models/bleeding_unet/bleeding_report.json).
A frame is flagged when >= min_frac of its pixels are predicted bleeding (the same rule as the evaluation).
Run every `every_s` seconds of video and held in between. Research prototype - NOT clinically validated.
"""

from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class BleedingDetector:
    def __init__(self, checkpoint, every_s: float = 0.5, min_frac: float = 0.002, device: str = "cpu",
                 threshold: float = 0.5):
        import segmentation_models_pytorch as smp
        import torch

        self.torch = torch
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.h, self.w = ck.get("input_hw", [288, 512])
        m = smp.Unet("resnet18", encoder_weights=None, classes=1)
        m.load_state_dict(ck["model_state"])
        self.device = device
        self.model = m.to(device).eval()
        self.every_s, self.min_frac, self.threshold = float(every_s), float(min_frac), float(threshold)
        self.last_t, self.mask = None, None

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        x = ((img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].copy()
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.model(self.torch.from_numpy(x).to(self.device)))[0, 0].cpu().numpy()
        return cv2.resize((p >= self.threshold).astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0

    def update(self, bgr: np.ndarray, t: float) -> dict:
        if self.mask is None or self.last_t is None or t - self.last_t >= self.every_s or t < self.last_t:
            self.mask, self.last_t = self.predict(bgr), t
        frac = float(self.mask.mean())
        return {"bleeding_frac": round(frac, 4), "bleeding_detected": bool(frac >= self.min_frac)}


def draw_bleeding(img: np.ndarray, mask: np.ndarray | None, detected: bool) -> np.ndarray:
    """Thin dark-red outline of the bleeding region + a small 'Bleeding' tag (keeps the overlay uncluttered)."""
    if mask is None or not detected:
        return img
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "Bleeding", (12, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2, cv2.LINE_AA)
    return img
