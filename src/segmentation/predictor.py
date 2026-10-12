"""Inference wrapper shared by instrument and tissue segmentation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.segmentation_data import IMAGENET_MEAN, IMAGENET_STD, normalize_image
from src.segmentation.models import build_model

# Distinct BGR colours for overlays (index 0 = background is never drawn).
PALETTE_BGR = [
    (0, 0, 0), (0, 255, 170), (184, 255, 169), (140, 140, 210), (114, 114, 255),
    (156, 70, 231), (75, 183, 186), (0, 85, 255), (0, 0, 255), (0, 255, 255),
    (165, 160, 255), (128, 50, 0), (0, 74, 111), (255, 0, 255), (255, 255, 0),
]


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train one with train_segmentation.py "
            f"or set the path in the config.")
    # Checkpoints hold only tensors and plain containers, so the safe loader works.
    ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "meta" not in ckpt:
        raise ValueError(
            f"{path} is not a PRISM-Endo checkpoint (missing 'model_state'/'meta'). "
            f"For the old notebook model use scripts/evaluate_legacy_unet.py.")
    return ckpt


class SegmentationPredictor:
    """Runs a trained segmentation model on full-resolution BGR frames and
    returns a label map (uint8, class indices) at the input resolution."""

    def __init__(self, model, meta: dict, device: str = "cpu"):
        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.meta = meta
        self.device = device
        self.class_names: list[str] = meta["class_names"]
        self.num_classes = len(self.class_names)
        self.input_h, self.input_w = meta["input_size"]
        self.norm = meta.get("normalization", "imagenet")
        self.threshold = float(meta.get("binary_threshold", 0.5))
        self.upsample = "bilinear"  # or "nearest" (set from the pipeline config)

    _cache: dict = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", cache: bool = True) -> "SegmentationPredictor":
        """Load a predictor. Loaded models are reused (same file, same modification time, same device), so a new
        pipeline with different cheap settings does not reload the weights."""
        p = Path(path)
        key = (str(p.resolve()), p.stat().st_mtime if p.is_file() else 0, str(device))
        if cache and key in cls._cache:
            return cls._cache[key]
        ckpt = load_checkpoint(path, "cpu")  # load on CPU, then move (needed for DirectML)
        meta = ckpt["meta"]
        out_ch = 1 if meta.get("single_logit") else len(meta["class_names"])
        model = build_model(meta["model"], out_ch, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        pred = cls(model, meta, device)
        if cache:
            cls._cache[key] = pred
        return pred

    def _prepare(self, bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        if self.norm == "imagenet":
            return normalize_image(small, IMAGENET_MEAN, IMAGENET_STD)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def use_onnx(self, onnx_path: str | Path, threads: int | None = None) -> None:
        """Run the network with ONNX Runtime on CPU (measured 1.8-2.0x faster than PyTorch on the
        test CPU with identical argmax labels; see scripts/export_onnx.py)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.ort = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def predict_logits(self, frames: list[np.ndarray]):
        torch = self.torch
        arr = np.stack([self._prepare(f) for f in frames])
        if getattr(self, "ort", None) is not None:
            return torch.from_numpy(self.ort.run(None, {"image": arr})[0])
        batch = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():  # inference_mode is not supported by every backend (DirectML)
            return self.model(batch)

    def predict(self, bgr: np.ndarray, return_probs: bool = False):
        return self.predict_batch([bgr], return_probs)[0]

    def predict_batch(self, frames: list[np.ndarray], return_probs: bool = False):
        torch = self.torch
        logits = self.predict_logits(frames)
        h, w = frames[0].shape[:2]
        if not return_probs and logits.shape[1] > 1:
            # Labels only: argmax(logits) == argmax(softmax(logits)), so skip the softmax.
            labels = self.labels_from_lowres(logits, (h, w))
            return [labels[i] for i in range(len(frames))]
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            labels = (probs[:, 0] > self.threshold).to(torch.uint8)
            probs = torch.cat([1 - probs, probs], 1)
        else:
            probs = torch.softmax(logits, 1)
            labels = probs.argmax(1).to(torch.uint8)
        labels = labels.cpu().numpy().astype(np.uint8)
        if return_probs:
            probs = probs.cpu().numpy()
            return [(labels[i], probs[i]) for i in range(len(frames))]
        return [labels[i] for i in range(len(frames))]

    def predict_lowres_probs(self, frames: list[np.ndarray]) -> np.ndarray:
        """Softmax probabilities at the model's input resolution, (B, C, h, w) float32 on CPU."""
        logits = self.predict_logits(frames)
        if logits.shape[1] == 1:
            p = self.torch.sigmoid(logits)
            return self.torch.cat([1 - p, p], 1).cpu().numpy()
        return self.torch.softmax(logits, 1).cpu().numpy()

    def labels_from_lowres(self, scores, out_hw: tuple[int, int], upsample: str | None = None) -> np.ndarray:
        """Turn low-resolution scores (logits or probabilities, (B, C, h, w), torch or numpy)
        into full-resolution label maps (B, H, W) uint8.

        upsample="bilinear": interpolate the C score maps to full size, then argmax (smooth boundaries).
        upsample="nearest" : argmax at model resolution, then nearest-neighbour resize (much cheaper on CPU).
        """
        torch = self.torch
        mode = upsample or self.upsample
        h, w = out_hw
        if mode == "nearest":
            if isinstance(scores, np.ndarray):
                low = scores.argmax(1).astype(np.uint8)
            else:
                low = scores.argmax(1).to(torch.uint8).cpu().numpy()
            return np.stack([cv2.resize(l, (w, h), interpolation=cv2.INTER_NEAREST) for l in low])
        t = torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        return t.argmax(1).to(torch.uint8).cpu().numpy()


def colorize(labels: np.ndarray, palette=PALETTE_BGR) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(palette):
        lut[i] = c
    return lut[labels]


def overlay(bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.45,
            class_ids: list[int] | None = None, palette=PALETTE_BGR) -> np.ndarray:
    """Alpha-blend coloured labels onto the frame (background untouched)."""
    color = colorize(labels, palette)
    sel = labels > 0 if class_ids is None else np.isin(labels, class_ids)
    out = bgr.copy()
    out[sel] = (bgr[sel] * (1 - alpha) + color[sel] * alpha).astype(np.uint8)
    return out
