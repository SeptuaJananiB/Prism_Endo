"""Train TipNet (src/tracking/tipnet.py) by backpropagation on the automatic PSEUDO tip labels of 16 CholecSeg8k videos.

Split by VIDEO: train = 13 videos, val = video20/26/43 (model selection on pseudo labels). video01 (which holds the 40
human-labelled evaluation frames) is never used. Final accuracy: scripts/eval_tipnet_human.py on the human frames.
Runs on the AMD GPU via DirectML (num_workers 0, SimpleAdamW), saves CPU-tensor checkpoints.

  .venv-dml\\Scripts\\python scripts\\train_tipnet.py --epochs 12
"""

import argparse
import json
import math
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
from src.tracking import tipnet as tn  # noqa: E402
from src.utils.device import is_directml, select_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VAL = {"video20", "video26", "video43"}


class TipData(Dataset):
    def __init__(self, df, train: bool, seed: int = 0, sigma: float = 5.0):
        self.df, self.train, self.rng, self.sigma = df.reset_index(drop=True), train, random.Random(seed), sigma

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(str(ROOT / r["image"]))
        pts = tn.labels_from_row(r)
        if self.train:
            img, pts = tn.augment(img, pts, self.rng)
        sx, sy = tn.IN_W / img.shape[1], tn.IN_H / img.shape[0]
        hm = tn.gaussian_heatmaps([[(x * sx, y * sy) for x, y in p] for p in pts], sigma=self.sigma)
        return torch.from_numpy(tn.preprocess(img)), torch.from_numpy(hm)


def match_errors(preds, gts):
    """identity-free nearest matching -> list of pixel errors (unmatched gts are counted by the caller)."""
    used, errs = set(), []
    for g in gts:
        c = [(math.dist(g, p[:2]), i) for i, p in enumerate(preds) if i not in used]
        if c:
            d, i = min(c)
            used.add(i)
            errs.append(d)
    return errs


@torch.no_grad()
def validate(model, df, device, thr=0.3):
    model.eval()
    stats = {c: {"err": [], "gt": 0, "fp": 0} for c in tn.CHANNELS}
    for i in range(0, len(df), 8):
        rows = df.iloc[i:i + 8]
        x = torch.stack([torch.from_numpy(tn.preprocess(cv2.imread(str(ROOT / r["image"])))) for _, r in rows.iterrows()])
        heat = torch.sigmoid(model(x.to(device))).cpu().numpy()
        for (_, r), h in zip(rows.iterrows(), heat):
            d = tn.decode(h, thr)
            for c, gts in zip(tn.CHANNELS, tn.labels_from_row(r)):
                e = match_errors(d[c], gts)
                stats[c]["err"] += e
                stats[c]["gt"] += len(gts)
                stats[c]["fp"] += max(0, len(d[c]) - len(gts))
    out = {}
    for c, s in stats.items():
        e = np.array(s["err"])
        out[c] = {"gt": s["gt"], "recall": len(e) / max(1, s["gt"]), "median_px": float(np.median(e)) if len(e) else None,
                  "pck20": float((e <= 20).mean()) if len(e) else 0.0, "extra_preds": s["fp"]}
    # selection score: PCK@20 x recall averaged over channels (rewards accurate AND complete detections)
    out["score"] = float(np.mean([out[c]["pck20"] * out[c]["recall"] for c in tn.CHANNELS]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=5.0, help="target Gaussian sigma in input px (input = 448x256)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/models/tipnet")
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: stop each epoch after N batches")
    ap.add_argument("--resume", default=None, help="continue from a checkpoint (e.g. outputs/models/tipnet/last.pt)")
    ap.add_argument("--pause-ms", type=int, default=0, help="sleep after each batch (cooler laptop, slower training)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (data loading/augmentation)")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "outputs" / "auto_labels" / "all_frames.csv")
    assert not (df.video == "video01").any(), "video01 must never be used for TipNet training"
    tr, va = df[~df.video.isin(VAL)], df[df.video.isin(VAL)]
    device = select_device(a.device)
    model = tn.build_model().to(device)
    opt = SimpleAdamW(model.parameters(), lr=a.lr, weight_decay=1e-4) if is_directml(device) else \
        torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loader = DataLoader(TipData(tr, True, sigma=a.sigma), batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"device {device}; train {len(tr)} frames ({tr.video.nunique()} videos), val {len(va)} frames", flush=True)
    best, hist, start = -1.0, [], 1
    if a.resume:
        ck = torch.load(ROOT / a.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state"])
        start = int(ck["epoch"]) + 1
        if (out / "history.csv").is_file():
            hist = pd.read_csv(out / "history.csv").to_dict("records")[: start - 1]
        if (out / "best.pt").is_file():
            best = float(torch.load(out / "best.pt", map_location="cpu", weights_only=True)["val"]["score"])
        for _ in range(start - 1):
            sched.step()
        print(f"resumed from {a.resume} at epoch {start} (best score {best:.3f})", flush=True)
    for ep in range(start, a.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(loader):
            if a.max_batches and bi >= a.max_batches:
                break
            loss = tn.focal_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss.detach().cpu()), nb + 1
            if a.pause_ms:
                time.sleep(a.pause_ms / 1000)
        sched.step()
        v = validate(model, va if not a.max_batches else va.iloc[:64], device)
        rec = {"epoch": ep, "train_loss": tot / max(1, nb), "sec": round(time.time() - t0), **{
            f"{c}_{k}": v[c][k] for c in tn.CHANNELS for k in ("recall", "median_px", "pck20")}, "score": v["score"]}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        state = {k: t.detach().cpu() for k, t in model.state_dict().items()}
        torch.save({"model_state": state, "epoch": ep, "val": v, "channels": tn.CHANNELS}, out / "last.pt")
        if v["score"] > best:
            best = v["score"]
            torch.save({"model_state": state, "epoch": ep, "val": v, "channels": tn.CHANNELS}, out / "best.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print(f"done; best val score {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
