"""Evaluate a trained segmentation checkpoint on a held-out split.

Reports dataset-level per-class Dice/IoU at full frame resolution, frame-level
instrument-presence precision/recall (how empty frames are handled), and
inference latency. Classes absent from the split's ground truth are reported
as not evaluable instead of being scored.

  python evaluate_segmentation.py --config configs/instrument_seg.yaml \
      --checkpoint outputs/models/instrument_seg/best.pt --split test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.data.segmentation_data import build_lut, discover_from_config, read_label, split_samples
from src.segmentation.metrics import ConfusionMatrix
from src.segmentation.predictor import SegmentationPredictor, overlay
from src.utils.config import load_config, resolve_path
from src.utils.device import describe_hardware, select_device


def presence_stats(pred_present: list[bool], gt_present: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred_present, gt_present))
    fp = sum(p and not g for p, g in zip(pred_present, gt_present))
    fn = sum(g and not p for p, g in zip(pred_present, gt_present))
    tn = sum((not p) and (not g) for p, g in zip(pred_present, gt_present))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "frames_without_gt_instance": tn + fp}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None, help="output folder (default: <checkpoint dir>/eval_<split>)")
    p.add_argument("--save-overlays", type=int, default=12, help="number of example overlays to save")
    p.add_argument("--min-area", type=int, default=200, help="px; a class counts as present above this")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    device = select_device(cfg.get("device", "auto"))
    pred = SegmentationPredictor.from_checkpoint(resolve_path(args.checkpoint), device)
    if pred.class_names != [c["name"] for c in cfg["classes"]]:
        print("WARNING: checkpoint classes differ from config; using checkpoint classes.")
    lut = build_lut(pred.meta["classes"], cfg["data"].get("ignore_values", [255]),
                    cfg["data"].get("unlisted", "background"))
    samples = discover_from_config(cfg["data"])   # data.root may be a list; data.group_by clip|video
    split = split_samples(samples, cfg["data"]["splits"]).get(args.split, [])
    if not split:
        print(f"Split '{args.split}' is empty; nothing to evaluate.")
        return 1

    out_dir = Path(args.out) if args.out else resolve_path(args.checkpoint).parent / f"eval_{args.split}"
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    names = pred.class_names
    cm = ConfusionMatrix(len(names))
    fg_ids = list(range(1, len(names)))
    pres_pred = {n: [] for n in names[1:]}
    pres_gt = {n: [] for n in names[1:]}
    overlay_every = max(1, len(split) // max(1, args.save_overlays))
    latencies = []

    for start in range(0, len(split), args.batch_size):
        chunk = split[start:start + args.batch_size]
        frames = [cv2.imread(str(s.image)) for s in chunk]
        t0 = time.perf_counter()
        labels = pred.predict_batch(frames)
        latencies.append((time.perf_counter() - t0) / len(chunk))
        for k, (s, img, lab) in enumerate(zip(chunk, frames, labels)):
            gt = read_label(s.label, lut)
            cm.update(lab, gt)
            for c in fg_ids:
                pres_pred[names[c]].append(int((lab == c).sum()) >= args.min_area)
                pres_gt[names[c]].append(int((gt == c).sum()) >= args.min_area)
            if (start + k) % overlay_every == 0:
                vis = np.concatenate([overlay(img, np.where(gt == 255, 0, gt)), overlay(img, lab)], 1)
                cv2.putText(vis, "ground truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, "prediction", (img.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.imwrite(str(out_dir / "overlays" / f"{s.group}_{s.image.stem}.jpg"), vis)

    summary = cm.summary(names)
    summary["presence"] = {n: presence_stats(pres_pred[n], pres_gt[n]) for n in names[1:]}
    summary["latency_ms_per_frame_mean"] = 1000 * float(np.mean(latencies[1:] or latencies))
    summary["checkpoint"] = str(args.checkpoint)
    summary["split"] = args.split
    summary["groups"] = sorted({s.group for s in split})
    summary["videos"] = sorted({s.video for s in split})
    summary["group_by"] = cfg["data"].get("group_by", "clip")
    summary["hardware"] = describe_hardware()
    summary["note"] = ("Metrics at full frame resolution. 'None' means the class is absent from both "
                       "ground truth and prediction in this split, so it cannot be evaluated.")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluated {summary['frames']} frames from {summary['groups']}")
    print(f"{'class':26s} {'Dice':>7s} {'IoU':>7s} {'GT px':>11s}")
    for n, r in summary["per_class"].items():
        fmt = lambda v: f"{v:7.4f}" if v is not None else "    n/a"
        note = "" if r["present_in_ground_truth"] else "  (absent from GT: not evaluable)"
        print(f"{n:26s} {fmt(r['dice'])} {fmt(r['iou'])} {r['gt_pixels']:11d}{note}")
    print(f"mean Dice (present fg classes): {summary['mean_dice_present_fg']}")
    print(f"latency: {summary['latency_ms_per_frame_mean']:.1f} ms/frame on {device}")
    print(f"Saved {out_dir / 'metrics.json'} and overlays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
