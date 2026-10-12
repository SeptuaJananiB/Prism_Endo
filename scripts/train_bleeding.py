"""Learned BLEEDING detector (adverse-event signal) from SurgBlood (surgeon-reviewed bleeding-region masks).

Data: SurgBlood public test release (Pei et al. 2025, github.com/PJLallen/SurgBlood; academic use only): 20 laparoscopic
clips, 886 frames, 1280x720, binary bleeding masks (770 frames with bleeding, 116 without). The split is by CLIP/patient
and fixed in advance (below). Model: U-Net ResNet-18 (ImageNet), BCE + soft Dice, AMD GPU (DirectML). Selection on
validation; test scored once: pixel Dice/IoU on test frames, frame-level detection (bleeding present if the predicted
region covers >= --min-frac of the frame): precision, recall, F1, AUROC of the max probability.
Domain note: hepatobiliary/laparoscopic videos from another hospital - transfer to CholecSeg8k is NOT validated.

  .venv-dml\\Scripts\\python scripts\\train_bleeding.py --epochs 25
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.segmentation.optim import SimpleAdamW  # noqa: E402
from src.utils.device import is_directml, select_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/surgblood/SurgBlood-Dataset-test/test"
H, W = 288, 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TEST = ["2024-05-13patient2", "2024-06-17patient20", "supple6", "supple9"]   # fixed before training
VAL = ["2024-05-17patient4", "2024-03-08patient16", "supple2"]


def clip_split(name: str) -> str:
    if any(name.startswith(t) for t in TEST):
        return "test"
    if any(name.startswith(v) for v in VAL):
        return "val"
    return "train"


def frames(split: str):
    out = []
    for c in sorted(p for p in (DATA / "videos-image").iterdir() if p.is_dir()):
        if clip_split(c.name) != split:
            continue
        for img in sorted(c.glob("*.jpg")):
            m = DATA / "videos-mask" / c.name / img.name
            if m.is_file():
                out.append((img, m, c.name))
    return out


class BleedData(Dataset):
    def __init__(self, items, train, seed=0):
        self.items, self.train, self.rng = items, train, random.Random(seed)
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def load(self, i):
        if i not in self.cache:
            img, m, _ = self.items[i]
            x = cv2.resize(cv2.imread(str(img)), (W, H), interpolation=cv2.INTER_AREA)
            y = (cv2.resize(cv2.imread(str(m), cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST) > 127)
            self.cache[i] = (x, y.astype(np.float32))
        return self.cache[i]

    def __getitem__(self, i):
        x, y = self.load(i)
        if self.train:
            if self.rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if self.rng.random() < 0.3:
                x, y = x[::-1], y[::-1]
            x = np.clip(x.astype(np.float32) * self.rng.uniform(0.8, 1.2) + self.rng.uniform(-15, 15), 0, 255)
        x = ((np.ascontiguousarray(x)[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(y)[None].copy())


def build():
    import segmentation_models_pytorch as smp

    return smp.Unet("resnet18", encoder_weights="imagenet", classes=1)


@torch.no_grad()
def predict(model, items, device):
    model.eval()
    probs = []
    for x, _ in DataLoader(BleedData(items, False), batch_size=8, num_workers=0):
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0])
    return np.concatenate(probs)


def evaluate(prob, items, min_frac):
    from sklearn.metrics import roc_auc_score

    gts = [(cv2.resize(cv2.imread(str(m), cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST) > 127)
           for _, m, _ in items]
    inter = union = tp_pix = gt_pix = pr_pix = 0
    dices = []
    det_true, det_pred, det_score = [], [], []
    for p, g in zip(prob, gts):
        pb = p >= 0.5
        if g.any():
            d = 2 * (pb & g).sum() / max(1, pb.sum() + g.sum())
            dices.append(d)
        inter += (pb & g).sum()
        union += (pb | g).sum()
        det_true.append(g.mean() >= 0.0005)
        det_pred.append(pb.mean() >= min_frac)
        det_score.append(float(p.max()))
    t, pr = np.array(det_true), np.array(det_pred)
    tp, fp, fn = int((t & pr).sum()), int((~t & pr).sum()), int((t & ~pr).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"frames": len(items), "frames_with_bleeding": int(t.sum()),
            "dice_mean_on_bleeding_frames": float(np.mean(dices)) if dices else None,
            "iou_dataset": float(inter / max(1, union)),
            "detection_precision": prec, "detection_recall": rec, "detection_f1": 2 * prec * rec / max(1e-9, prec + rec),
            "detection_auroc": float(roc_auc_score(t, det_score)) if 0 < t.sum() < len(t) else None,
            "false_alarms_on_no_bleeding_frames": fp, "no_bleeding_frames": int((~t).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-frac", type=float, default=0.002)
    ap.add_argument("--out", default="outputs/models/bleeding_unet")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = frames("train"), frames("val"), frames("test")
    print(f"frames train {len(tr)} ({len({c for *_, c in tr})} clips), val {len(va)}, test {len(te)}", flush=True)
    device = select_device("auto")
    model = build().to(device)
    opt = SimpleAdamW(model.parameters(), lr=a.lr, weight_decay=1e-4) if is_directml(device) else \
        torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loader = DataLoader(BleedData(tr, True), batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)
    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = torch.sigmoid(model(x)).clamp(1e-5, 1 - 1e-5)
            bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = bce + dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss.detach().cpu()), nb + 1
        sched.step()
        vm = evaluate(predict(model, va, device), va, a.min_frac)
        score = (vm["dice_mean_on_bleeding_frames"] or 0) + vm["detection_f1"]
        rec = {"epoch": ep, "train_loss": round(tot / nb, 4), "val_dice": round(vm["dice_mean_on_bleeding_frames"] or 0, 4),
               "val_det_f1": round(vm["detection_f1"], 4), "sec": round(time.time() - t0)}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if score > best:
            best = score
            torch.save({"model_state": state, "epoch": ep, "val": vm, "input_hw": [H, W]}, out / "best.pt")
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state"])
    tm = evaluate(predict(model, te, device), te, a.min_frac)
    rep = {"data": "SurgBlood public test release, split by clip (fixed in advance)", "test_clips": TEST, "val_clips": VAL,
           "selected_epoch_on_val": ck["epoch"], "val": ck["val"], "test": tm, "history": hist,
           "note": "Detects visible bleeding (adverse event) - not a prediction of injury; transfer to CholecSeg8k "
                   "videos not validated."}
    (out / "bleeding_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"test": tm}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
