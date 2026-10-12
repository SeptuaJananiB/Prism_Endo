"""Training / evaluation loops for instrument and tissue segmentation.

Both tasks share this code; the task config decides the classes. Training
never starts implicitly: it only runs via train_segmentation.py.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from src.data.segmentation_data import (IGNORE_INDEX, SegmentationDataset, build_lut,
                                        class_names, discover_from_config, split_samples)
from src.segmentation.metrics import ConfusionMatrix
from src.segmentation.models import build_model
from src.utils.config import resolve_path
from src.utils.device import get_logger, is_directml, select_device

log = get_logger("prism.seg.train")


def make_datasets(cfg: dict, splits_wanted=("train", "val", "test")) -> tuple[dict, list[str], np.ndarray]:
    data_cfg = cfg["data"]
    classes = cfg["classes"]
    lut = build_lut(classes, data_cfg.get("ignore_values", [255]), data_cfg.get("unlisted", "background"))
    # data.root: one folder or a list; data.group_by: clip (default) | video
    samples = discover_from_config(data_cfg)
    split = split_samples(samples, data_cfg["splits"])
    size = tuple(cfg["input_size"])
    out = {}
    for name in splits_wanted:
        subset = split.get(name, [])
        limit = cfg.get("train", {}).get(f"limit_{name}_samples")
        if limit:
            # Evenly spaced subset so every group stays represented.
            idx = np.linspace(0, len(subset) - 1, num=min(limit, len(subset))).astype(int)
            subset = [subset[i] for i in sorted(set(idx.tolist()))]
        aug = cfg.get("augment") if name == "train" else None
        out[name] = SegmentationDataset(subset, lut, size, aug, cache=cfg.get("train", {}).get("cache", True),
                                        seed=cfg.get("seed", 0))
        log.info("%s: %d frames from groups %s (videos %s)", name, len(subset),
                 sorted({s.group for s in subset}), sorted({s.video for s in subset}))
    return out, class_names(classes), lut


def soft_dice_loss(logits, target, num_classes: int, ignore_index: int = IGNORE_INDEX, eps: float = 1.0):
    """Multi-class soft Dice loss using only float ops (works on CUDA, CPU
    and DirectML). Ignored pixels are excluded; classes absent from the batch
    ground truth are skipped (same behaviour as smp's DiceLoss)."""
    import torch

    valid = (target != ignore_index).unsqueeze(1).float()
    tgt = torch.where(target == ignore_index, torch.zeros_like(target), target)
    classes = torch.arange(num_classes, device=target.device).view(1, -1, 1, 1)
    onehot = (tgt.unsqueeze(1) == classes).float() * valid
    probs = torch.softmax(logits, 1) * valid
    inter = (probs * onehot).sum((0, 2, 3))
    card = (probs + onehot).sum((0, 2, 3))
    dice = (2 * inter + eps) / (card + eps)
    present = (onehot.sum((0, 2, 3)) > 0).float()
    return ((1 - dice) * present).sum() / present.sum().clamp(min=1.0)


def _loss_fn(cfg: dict, num_classes: int, device: str):
    import torch

    weights = cfg.get("train", {}).get("class_weights")
    ce = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device) if weights else None,
        ignore_index=IGNORE_INDEX)
    dice = lambda logits, target: soft_dice_loss(logits, target, num_classes)  # noqa: E731
    dice_w = float(cfg.get("train", {}).get("dice_weight", 1.0))

    def loss(logits, target):
        return ce(logits, target) + dice_w * dice(logits, target)
    return loss


def evaluate_model(model, loader, num_classes: int, device: str) -> ConfusionMatrix:
    import torch

    cm = ConfusionMatrix(num_classes)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images.to(device)).argmax(1).cpu().numpy()
            for p, t in zip(preds, targets.numpy()):
                cm.update(p, t)
    return cm


def checkpoint_meta(cfg: dict, names: list[str]) -> dict:
    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False  # weights are in the checkpoint; never re-download
    return {"task": cfg.get("task", "segmentation"), "class_names": names,
            "classes": cfg["classes"], "input_size": list(cfg["input_size"]),
            "normalization": "imagenet", "model": model_cfg,
            "splits": cfg["data"]["splits"]}


def train(cfg: dict, resume: str | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    tcfg = cfg.get("train", {})
    torch.manual_seed(cfg.get("seed", 0))
    if tcfg.get("num_threads"):
        torch.set_num_threads(int(tcfg["num_threads"]))
    device = select_device(cfg.get("device", "auto"))
    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, names, _ = make_datasets(cfg, ("train", "val"))
    if len(datasets["train"]) == 0:
        raise ValueError("Training split is empty - check data.splits in the config.")
    nw = int(tcfg.get("num_workers", 0))
    if nw and is_directml(device):
        # Batches from worker processes (shared memory) triggered DirectML
        # "allocator out of memory" / device-removed errors on an RX 6800M.
        log.warning("num_workers=%d is not supported with DirectML here; using 0.", nw)
        nw = 0
    train_loader = DataLoader(datasets["train"], batch_size=int(tcfg.get("batch_size", 8)),
                              shuffle=True, num_workers=nw, drop_last=len(datasets["train"]) > tcfg.get("batch_size", 8))
    val_loader = DataLoader(datasets["val"], batch_size=int(tcfg.get("batch_size", 8)), num_workers=nw)

    model = build_model(cfg["model"], len(names)).to(device)
    lr, wd = float(tcfg.get("lr", 3e-4)), float(tcfg.get("weight_decay", 1e-4))
    if is_directml(device):
        # torch's AdamW uses lerp_, which DirectML executes on the CPU (slow).
        from src.segmentation.optim import SimpleAdamW

        optim = SimpleAdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(tcfg.get("epochs", 10))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    loss_fn = _loss_fn(cfg, len(names), device)
    meta = checkpoint_meta(cfg, names)

    start_epoch, best, history = 0, -math.inf, []
    if resume:
        ck = torch.load(str(resolve_path(resume)), map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            optim.load_state_dict(ck["optimizer_state"])
        if "scheduler_state" in ck:
            sched.load_state_dict(ck["scheduler_state"])
        start_epoch = int(ck.get("epoch", 0))
        best = float(ck.get("best_score", -math.inf))
        history = list(ck.get("history", []))
        log.info("Resumed from %s at epoch %d (best=%.4f)", resume, start_epoch, best)

    max_batches = tcfg.get("max_train_batches")
    log.info("Training on %s for epochs %d..%d (%d train / %d val frames, input %s)",
             device, start_epoch + 1, epochs, len(datasets["train"]), len(datasets["val"]), cfg["input_size"])

    # Gradient accumulation: effective batch = batch_size * accum (small batches save GPU/commit memory).
    accum = max(1, int(tcfg.get("grad_accum_steps", 1)))
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        optim.zero_grad(set_to_none=True)
        for bi, (images, targets) in enumerate(train_loader):
            if max_batches and bi >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            loss = loss_fn(model(images), targets)
            (loss / accum).backward()
            if (bi + 1) % accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
            run_loss += float(loss.item())
            nb += 1
        if nb % accum:  # flush a partial accumulation at the end of the epoch
            optim.step()
            optim.zero_grad(set_to_none=True)
        sched.step()
        train_time = time.time() - t0

        val_summary = {}
        score = -run_loss / max(nb, 1)  # fallback if no val split
        if len(datasets["val"]):
            cm = evaluate_model(model, val_loader, len(names), device)
            val_summary = cm.summary(names)
            if val_summary["mean_dice_present_fg"] is not None:
                score = val_summary["mean_dice_present_fg"]
        row = {"epoch": epoch + 1, "train_loss": run_loss / max(nb, 1), "train_seconds": round(train_time, 1),
               "val_mean_dice": val_summary.get("mean_dice_present_fg"),
               "val_mean_iou": val_summary.get("mean_iou_present_fg"),
               "lr": optim.param_groups[0]["lr"]}
        for n, r in val_summary.get("per_class", {}).items():
            row[f"val_dice_{n}"] = r["dice"]
        history.append(row)
        log.info("epoch %d/%d loss=%.4f val_mean_dice=%s (%.0fs)", epoch + 1, epochs, row["train_loss"],
                 f"{row['val_mean_dice']:.4f}" if row["val_mean_dice"] is not None else "n/a", train_time)

        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        state = {"model_state": cpu_state, "optimizer_state": _to_cpu(optim.state_dict()),
                 "scheduler_state": sched.state_dict(), "epoch": epoch + 1, "meta": meta,
                 "history": history, "best_score": max(best, score)}
        torch.save(state, out_dir / "last.pt")
        if score > best:
            best = score
            torch.save({"model_state": cpu_state, "epoch": epoch + 1, "meta": meta,
                        "history": history, "best_score": best}, out_dir / "best.pt")
            log.info("  new best (%.4f) -> %s", best, out_dir / "best.pt")
        del state, cpu_state  # don't hold ~170 MB of host copies through the next epoch
        _write_history(out_dir / "history.csv", history)

    result = {"output_dir": str(out_dir), "best_score": best, "epochs_completed": len(history),
              "device": device, "history": history}
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _to_cpu(obj):
    """Recursively move tensors to CPU so checkpoints load on any device."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _write_history(path: Path, history: list[dict]) -> None:
    keys: list[str] = []
    for row in history:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
