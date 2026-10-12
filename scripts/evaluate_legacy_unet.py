"""Measure what the legacy notebook model (models/unet_model.pth) learned.

02_Instrument_Segmentation.ipynb built its binary target as
`_endo_mask.png > 0`. In CholecSeg8k that file is a sparse annotation-tool
mask, so `> 0` means "this pixel was annotated with ANY class", not
"instrument". This script scores the legacy model against both targets:

  * instrument   = watershed label in {31 grasper, 32 L-hook}
  * any_annotated = _endo_mask.png > 0   (the notebook's actual target)

Caveat: the notebook used a random frame-level split (seed 42), so the
frames scored here may have been in its training set; treat the numbers as
a diagnostic of *what* it learned, not as a held-out accuracy.

  python scripts/evaluate_legacy_unet.py --weights models/unet_model.pth --size 128
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.segmentation_data import discover_from_config, split_samples  # noqa: E402
from src.segmentation.metrics import binary_dice_iou  # noqa: E402
from src.segmentation.models import LegacyUNet  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="models/unet_model.pth")
    p.add_argument("--config", default="configs/instrument_seg.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--size", type=int, default=128, help="square input size used by the notebook")
    p.add_argument("--every", type=int, default=4, help="evaluate every Nth frame")
    p.add_argument("--out", default="outputs/legacy_unet_eval")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = LegacyUNet(1)
    # Plain state_dict of tensors: the safe loader is sufficient.
    model.load_state_dict(torch.load(str(resolve_path(args.weights)), map_location="cpu", weights_only=True))
    model.eval()

    samples = discover_from_config(cfg["data"])   # data.root may be a list of folders
    split = split_samples(samples, cfg["data"]["splits"])[args.split][::args.every]
    tot = {"instrument": np.zeros(3), "any_annotated": np.zeros(3)}  # inter, pred, gt
    per_frame = {"instrument": [], "any_annotated": []}
    pred_frac = []
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(split):
        img = cv2.imread(str(s.image))
        rgb = cv2.cvtColor(cv2.resize(img, (args.size, args.size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
        with torch.inference_mode():
            logit = model(x)[0, 0].numpy()
        pred = cv2.resize((logit > 0).astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        ws = cv2.imread(str(s.label), cv2.IMREAD_GRAYSCALE)
        sparse = cv2.imread(str(s.image).replace("_endo.png", "_endo_mask.png"), cv2.IMREAD_GRAYSCALE)
        targets = {"instrument": np.isin(ws, [31, 32]), "any_annotated": sparse > 0}
        pred_frac.append(pred.mean())
        for k, t in targets.items():
            tot[k] += [np.logical_and(pred, t).sum(), pred.sum(), t.sum()]
            per_frame[k].append(binary_dice_iou(pred, t)[0])
        if i < 4:
            vis = img.copy()
            vis[pred] = (0.5 * vis[pred] + [0, 127, 0]).astype(np.uint8)
            cv2.imwrite(str(out / f"{s.group}_{s.image.stem}_legacy_pred.jpg"), vis)

    res = {"frames": len(split), "split": args.split, "input_size": args.size,
           "mean_predicted_foreground_fraction": float(np.mean(pred_frac))}
    for k, (inter, ps, gs) in tot.items():
        res[f"dice_vs_{k}"] = float(2 * inter / (ps + gs)) if ps + gs else None
        res[f"gt_fraction_{k}"] = float(gs / (len(split) * 480 * 854))
    with open(out / f"metrics_size{args.size}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
