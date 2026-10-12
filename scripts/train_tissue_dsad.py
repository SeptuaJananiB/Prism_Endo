"""Fine-tune the CholecSeg8k tissue model with DSAD (Dresden Surgical Anatomy Dataset) to cut false "liver" on
non-cholecystectomy video (e.g. appendicectomy: bowel/abdominal wall were called liver).

Start: outputs/models/tissue_seg_multivideo/best.pt (12 classes, unchanged head -> pipeline-compatible).
Train: CholecSeg8k train videos (full labels) + DSAD train (partial labels, unlabelled pixels ignored;
       scripts/prepare_dsad.py). Validation: CholecSeg8k val videos + DSAD validation (surgery split).
Model selection: mean of (CholecSeg8k val mean Dice, DSAD val mean Dice over abdominal_wall/liver/GI).
Test (once, after training): CholecSeg8k test videos 17/25/28 and DSAD test. Output: outputs/models/tissue_seg_dsad/.

  .venv-dml/Scripts/python scripts/train_tissue_dsad.py [--epochs 12]
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.segmentation_data import augment, normalize_image  # noqa: E402
from src.segmentation.metrics import ConfusionMatrix  # noqa: E402
from src.segmentation.models import build_model  # noqa: E402
from src.segmentation.train import evaluate_model, make_datasets, soft_dice_loss  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import is_directml, select_device  # noqa: E402

DSAD_CLASSES = ["abdominal_wall", "liver", "gastrointestinal_tract"]


class Tagged:
    """Wraps a dataset and appends a source flag (1 = DSAD) to each sample."""

    def __init__(self, ds, flag):
        self.ds, self.flag = ds, flag

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, self.flag


class DSADDataset:
    def __init__(self, split, aug_cfg=None, seed=0, root=None):
        d = Path(root or ROOT / "data/dsad/png") / split
        self.items = sorted(d.glob("*_img.*"))
        self.aug, self.rng = aug_cfg, random.Random(seed)
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        if i not in self.cache:
            p = self.items[i]
            self.cache[i] = (cv2.imread(str(p)), cv2.imread(str(p.with_name(p.name.split("_img.")[0] + "_lab.png")), 0))
        img, lab = self.cache[i]
        if self.aug:
            img, lab = augment(img, lab, self.rng, self.aug)
        return torch.from_numpy(normalize_image(img)), torch.from_numpy(lab.astype(np.int64))


def dsad_dice(cm: ConfusionMatrix, names):
    pc = cm.summary(names)["per_class"]
    d = {n: pc[n]["dice"] for n in DSAD_CLASSES}
    v = [x for x in d.values() if x is not None]
    return (float(np.mean(v)) if v else None), d


def main():
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--init", default="outputs/models/tissue_seg_multivideo/best.pt")
    ap.add_argument("--out", default="outputs/models/tissue_seg_dsad")
    ap.add_argument("--cholec-png", default=None,
                    help="pre-packed CholecSeg8k folder ({train,val,test}/*_img.png + class-index *_lab.png), e.g. on Colab")
    ap.add_argument("--dsad-weight", type=float, default=1.0, help="loss weight of DSAD batches (CholecSeg8k = 1)")
    ap.add_argument("--max-cholec-drop", type=float, default=None,
                    help="only select epochs whose CholecSeg8k val Dice >= original - this (keeps gallbladder quality)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/best.pt (weights + history)")
    a = ap.parse_args()
    torch.manual_seed(0)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/tissue_seg_multivideo.yaml")
    ck = torch.load(str(ROOT / a.init), map_location="cpu", weights_only=True)
    meta = dict(ck["meta"])
    names = meta["class_names"]
    C = len(names)
    if a.cholec_png:
        cs = {k: DSADDataset(k, cfg.get("augment") if k == "train" else None, root=a.cholec_png)
              for k in ("train", "val", "test")}
    else:
        cs, _, _ = make_datasets(cfg, ("train", "val", "test"))
    aug = cfg.get("augment")
    ds_tr, ds_va, ds_te = DSADDataset("train", aug), DSADDataset("validation"), DSADDataset("test")
    print(f"cholec train {len(cs['train'])} val {len(cs['val'])} test {len(cs['test'])} | "
          f"dsad train {len(ds_tr)} val {len(ds_va)} test {len(ds_te)}", flush=True)
    dev = select_device("auto")
    model = build_model(meta["model"], C, pretrained=False)
    model.load_state_dict(ck["model_state"])
    start, rk = 1, None
    if a.resume and (out / "best.pt").is_file():  # optimizer state is not saved: fresh AdamW, lr from the cosine schedule
        rk = torch.load(str(out / "best.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(rk["model_state"])
        start = int(rk["epoch"]) + 1
        print(f"resumed weights of epoch {rk['epoch']} from {out / 'best.pt'}", flush=True)
    model.to(dev)
    if is_directml(dev):
        from src.segmentation.optim import SimpleAdamW
        opt = SimpleAdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    w = cfg.get("train", {}).get("class_weights")
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev) if w else None,
                                   ignore_index=255)
    loss_fn = lambda lo, t: ce(lo, t) + soft_dice_loss(lo, t, C)  # noqa: E731
    tl = DataLoader(ConcatDataset([Tagged(cs["train"], 0), Tagged(ds_tr, 1)]), batch_size=a.bs, shuffle=True, drop_last=True)
    vl_c = DataLoader(cs["val"], batch_size=a.bs)
    vl_d = DataLoader(ds_va, batch_size=a.bs)

    def val():
        cc = evaluate_model(model, vl_c, C, dev).summary(names)["mean_dice_present_fg"]
        dd, per = dsad_dice(evaluate_model(model, vl_d, C, dev), names)
        return cc, dd, per

    if rk is not None:
        hist = [h for h in json.loads((out / "history.json").read_text()) if h["epoch"] < start]
        best = float(rk["best_score"])
        for _ in range(start - 1):
            sched.step()
    else:
        c0, d0, p0 = val()
        hist = [{"epoch": 0, "cholec_val_dice": c0, "dsad_val_dice": d0, **{f"dsad_{k}": v for k, v in p0.items()}}]
        print("epoch 0 (init)", hist[-1], flush=True)
        best = -math.inf
    meta["dsad_finetune"] = {"init": a.init, "dsad_mapping": "liver->liver; colon/small_intestine/stomach->GI; "
                             "abdominal_wall->abdominal_wall; rest ignored", "license": "DSAD CC BY 4.0"}
    for ep in range(start, a.epochs + 1):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for x, y, src in tl:
            x, y = x.to(dev), y.to(dev)
            lo = model(x)
            m = src.bool()
            loss = 0.0
            for sel, w in ((~m, 1.0), (m, a.dsad_weight)):  # index_select: DirectML-safe (no bool-mask indexing)
                if sel.any():
                    ix = torch.nonzero(sel).squeeze(1).to(dev)
                    loss = loss + w * loss_fn(lo.index_select(0, ix), y.index_select(0, ix)) * float(sel.float().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.item())
            nb += 1
        sched.step()
        c, d, per = val()
        score = (c + d) / 2
        if a.max_cholec_drop is not None:  # gallbladder guard: DSAD gain only counts if CholecSeg8k holds
            score = d if c >= hist[0]["cholec_val_dice"] - a.max_cholec_drop else -1 + c
        row = {"epoch": ep, "loss": run / nb, "cholec_val_dice": c, "dsad_val_dice": d,
               **{f"dsad_{k}": v for k, v in per.items()}, "sec": round(time.time() - t0)}
        hist.append(row)
        print(f"epoch {ep}/{a.epochs}", row, flush=True)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if score > best:
            best = score
            torch.save({"model_state": sd, "epoch": ep, "meta": meta, "history": hist, "best_score": best},
                       out / "best.pt")
            print("  new best", round(best, 4), flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
    # one-time test with the selected checkpoint, and the original model for comparison
    res = {}
    for tag, path in (("original", ROOT / a.init), ("dsad_finetuned", out / "best.pt")):
        model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True)["model_state"])
        model.to(dev)
        cc = evaluate_model(model, DataLoader(cs["test"], batch_size=a.bs), C, dev).summary(names)
        dd, per = dsad_dice(evaluate_model(model, DataLoader(ds_te, batch_size=a.bs), C, dev), names)
        res[tag] = {"cholec_test_mean_dice": cc["mean_dice_present_fg"],
                    "cholec_test_liver_dice": cc["per_class"]["liver"]["dice"],
                    "dsad_test_mean_dice": dd, "dsad_test_per_class": per}
        print("TEST", tag, res[tag], flush=True)
    (out / "test_results.json").write_text(json.dumps(res, indent=2))
    print("all done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
