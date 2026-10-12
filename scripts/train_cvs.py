"""Learned SAFETY model: Critical View of Safety (CVS) criteria C1/C2/C3 from Endoscapes-CVS201 (surgeon labels).

CVS is the safety check surgeons must reach before clipping/cutting in cholecystectomy (bile-duct-injury prevention).
Labels: average of 3 expert annotators per criterion (soft targets 0..1), frames every 5 s. Official VIDEO-level split:
120 train / 41 val / 40 test videos (data/endoscapes/endoscapes/*_vids.txt). Model selection on val; the test split is
scored once at the end with the standard metric (mAP over C1-C3, ground truth binarised at >= 0.5).
Runs on the AMD GPU (DirectML; num_workers 0, SimpleAdamW). It answers "is CVS achieved?", not tissue-injury risk.

  .venv-dml\\Scripts\\python scripts\\train_cvs.py --epochs 15
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.segmentation.optim import SimpleAdamW  # noqa: E402
from src.utils.device import is_directml, select_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "endoscapes" / "endoscapes"
H, W = 240, 432
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def split_frames(split: str) -> pd.DataFrame:
    vids = {int(float(v)) for v in (DATA / f"{split}_vids.txt").read_text().split()}
    m = pd.read_csv(DATA / "all_metadata.csv")
    m = m[m["vid"].isin(vids) & m["is_ds_keyframe"].astype(str).str.lower().eq("true")].copy()
    m["path"] = [DATA / split / f"{v}_{f}.jpg" for v, f in zip(m["vid"], m["frame"])]
    return m[m["path"].map(Path.is_file)].reset_index(drop=True)


class CVSData(Dataset):
    def __init__(self, df, train, seed=0):
        self.df, self.train, self.rng = df, train, random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(str(r["path"]))
        if self.train:
            h, w = img.shape[:2]
            s = self.rng.uniform(0.85, 1.0)
            ch, cw = int(h * s), int(w * s)
            y0, x0 = self.rng.randint(0, h - ch), self.rng.randint(0, w - cw)
            img = img[y0:y0 + ch, x0:x0 + cw]
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
            img = np.clip(img.astype(np.float32) * self.rng.uniform(0.8, 1.2) + self.rng.uniform(-20, 20), 0, 255)
        img = cv2.resize(np.ascontiguousarray(img).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
        x = ((img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        y = np.array([r["C1"], r["C2"], r["C3"]], np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def build_model():
    import torchvision

    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Linear(m.fc.in_features, 3)
    return m


@torch.no_grad()
def predict(model, df, device, bs=16):
    model.eval()
    out = []
    loader = DataLoader(CVSData(df, False), batch_size=bs, num_workers=0)
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)


def metrics(prob, df) -> dict:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score

    gt = (df[["C1", "C2", "C3"]].to_numpy() >= 0.5).astype(int)
    res = {}
    for k, c in enumerate(("C1", "C2", "C3")):
        if gt[:, k].min() == gt[:, k].max():
            res[c] = {"ap": None, "bal_acc": None, "positives": int(gt[:, k].sum())}
            continue
        res[c] = {"ap": float(average_precision_score(gt[:, k], prob[:, k])),
                  "bal_acc": float(balanced_accuracy_score(gt[:, k], prob[:, k] >= 0.5)),
                  "positives": int(gt[:, k].sum())}
    aps = [v["ap"] for v in res.values() if v["ap"] is not None]
    res["mAP"] = float(np.mean(aps)) if aps else None
    cvs_gt = gt.all(1)
    res["full_cvs_frames"] = int(cvs_gt.sum())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/models/cvs_resnet50")
    ap.add_argument("--pause-ms", type=int, default=50)
    ap.add_argument("--max-batches", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = split_frames("train"), split_frames("val"), split_frames("test")
    print(f"frames: train {len(tr)} ({tr['vid'].nunique()} videos), val {len(va)}, test {len(te)}", flush=True)
    device = select_device("auto")
    model = build_model().to(device)
    opt = SimpleAdamW(model.parameters(), lr=a.lr, weight_decay=1e-4) if is_directml(device) else \
        torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    pos = (tr[["C1", "C2", "C3"]].to_numpy() >= 0.5).mean(0)
    pos_weight = torch.tensor((1 - pos) / np.maximum(pos, 1e-3), dtype=torch.float32).clamp(max=10).to(device)
    loader = DataLoader(CVSData(tr, True), batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)
    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(loader):
            if a.max_batches and bi >= a.max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # BCE with logits written with basic ops (DirectML friendly), soft targets, positive re-weighting
            p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
            loss = -(pos_weight * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss.detach().cpu()), nb + 1
            if a.pause_ms:
                time.sleep(a.pause_ms / 1000)
        sched.step()
        vm = metrics(predict(model, va if not a.max_batches else va.iloc[:64], device), va if not a.max_batches else va.iloc[:64])
        rec = {"epoch": ep, "train_loss": round(tot / max(1, nb), 4), "val_mAP": vm["mAP"], "sec": round(time.time() - t0)}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if vm["mAP"] is not None and vm["mAP"] > best:
            best = vm["mAP"]
            torch.save({"model_state": state, "epoch": ep, "val": vm}, out / "best.pt")
        torch.save({"model_state": state, "epoch": ep, "val": vm}, out / "last.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    if a.max_batches:
        return 0
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state"])
    tm = metrics(predict(model, te, device), te)
    report = {"selected_epoch_on_val": ck["epoch"], "val": ck["val"], "test": tm,
              "note": "Endoscapes-CVS201 official video split; test scored once with the val-selected model. "
                      "Ground truth = mean of 3 surgeons, binarised at 0.5."}
    (out / "cvs_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"test_mAP": tm["mAP"], **{c: tm[c] for c in ("C1", "C2", "C3")}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
