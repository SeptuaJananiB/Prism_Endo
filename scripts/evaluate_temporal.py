"""Measure mask flicker vs. accuracy for temporal smoothing variants on held-out clips.

For every test clip (consecutive 25 fps frames) both models are run once per
frame; the low-resolution probabilities are cached and then re-used for each
variant (EMA alpha, with/without camera-motion compensation, bilinear vs
nearest upsampling). Reported per model and variant:

  mean Dice (present foreground classes, dataset level, full resolution)
  pred_change   - fraction of valid pixels whose predicted label changes between consecutive frames
  gt_change     - same for the ground truth (real motion; the floor for a perfect model)
  spurious      - fraction of pixels whose GT label did NOT change but whose prediction did (flicker)

  python scripts/evaluate_temporal.py --split test
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.data.segmentation_data import build_lut, discover_samples, read_label, split_samples  # noqa: E402
from src.motion.camera import CameraMotionEstimator  # noqa: E402
from src.segmentation.metrics import ConfusionMatrix  # noqa: E402
from src.segmentation.predictor import SegmentationPredictor  # noqa: E402
from src.segmentation.temporal import TemporalSmoother  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.device import describe_hardware  # noqa: E402

VARIANTS = [  # (name, alpha, motion_compensate, upsample)
    ("none", 1.0, False, "bilinear"),
    ("none_nearest_upsample", 1.0, False, "nearest"),
    ("ema0.7", 0.7, False, "bilinear"),
    ("ema0.7_mc", 0.7, True, "bilinear"),
    ("ema0.5", 0.5, False, "bilinear"),
    ("ema0.5_mc", 0.5, True, "bilinear"),
    ("ema0.35_mc", 0.35, True, "bilinear"),
]


def frame_no(p: Path) -> int:
    return int(re.search(r"frame_(\d+)", p.name).group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/seg_base.yaml")
    ap.add_argument("--instrument", default="outputs/models/instrument_seg/best.pt")
    ap.add_argument("--tissue", default="outputs/models/tissue_seg/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", default="outputs/eval_temporal")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    models = {"instrument": SegmentationPredictor.from_checkpoint(resolve_path(args.instrument), "cpu"),
              "tissue": SegmentationPredictor.from_checkpoint(resolve_path(args.tissue), "cpu")}
    luts = {k: build_lut(m.meta["classes"], cfg["data"].get("ignore_values", [255]),
                         cfg["data"].get("unlisted", "background")) for k, m in models.items()}
    samples = discover_samples(resolve_path(cfg["data"]["root"]), cfg["data"].get("layout", "cholecseg8k"),
                               cfg["data"].get("label_suffix", "_endo_watershed_mask.png"))
    split = split_samples(samples, cfg["data"]["splits"])[args.split]
    clips: dict[str, list] = {}
    for s in split:
        clips.setdefault(s.image.parent.name, []).append(s)

    stats = {k: {v[0]: {"cm": ConfusionMatrix(len(m.class_names)), "pc": 0.0, "gc": 0.0, "sp": 0.0, "n": 0.0}
                 for v in VARIANTS} for k, m in models.items()}
    cam_log = []
    t_start = time.time()
    for clip, items in sorted(clips.items()):
        items = sorted(items, key=lambda s: frame_no(s.image))
        frames = [cv2.imread(str(s.image)) for s in items]
        times = [(frame_no(s.image) - frame_no(items[0].image)) / args.fps for s in items]
        h, w = frames[0].shape[:2]
        gts = {k: np.stack([read_label(s.label, luts[k]) for s in items]) for k in models}
        probs = {k: np.concatenate([m.predict_lowres_probs(frames[i:i + 4]) for i in range(0, len(frames), 4)])
                 .astype(np.float16) for k, m in models.items()}
        # Camera motion, excluding (unsmoothed) predicted instrument pixels.
        est = CameraMotionEstimator()
        inst_lab = models["instrument"].labels_from_lowres(probs["instrument"].astype(np.float32), (h, w), "nearest")
        cams = [est.update(f, inst_lab[i] > 0) for i, f in enumerate(frames)]
        cam_log += [{"clip": clip, "ok": c.ok, "tx": c.translation[0], "ty": c.translation[1],
                     "rot": c.rotation_deg, "scale": c.scale, "inlier_ratio": c.inlier_ratio,
                     "residual_px": c.residual_px, "ms": c.ms} for c in cams[1:]]
        for k, m in models.items():
            gt = gts[k]
            for name, alpha, mc, up in VARIANTS:
                sm = TemporalSmoother(alpha, mc)
                prev = None
                st = stats[k][name]
                for i in range(len(items)):
                    p = sm.update(probs[k][i].astype(np.float32), times[i], (h, w), cams[i])
                    lab = m.labels_from_lowres(p[None], (h, w), up)[0]
                    st["cm"].update(lab, gt[i])
                    if prev is not None:
                        valid = (gt[i] != 255) & (gt[i - 1] != 255)
                        nv = valid.sum()
                        pc = (lab != prev) & valid
                        gc = (gt[i] != gt[i - 1]) & valid
                        st["pc"] += pc.sum() / nv
                        st["gc"] += gc.sum() / nv
                        st["sp"] += (pc & ~gc).sum() / nv
                        st["n"] += 1
                    prev = lab
        print(f"{clip}: done ({time.time() - t_start:.0f}s)", flush=True)

    out = {"split": args.split, "clips": sorted(clips), "hardware": describe_hardware(), "models": {}}
    for k, m in models.items():
        out["models"][k] = {}
        print(f"\n== {k} ==\n{'variant':24s} {'meanDice':>8s} {'pred_chg%':>9s} {'gt_chg%':>8s} {'spurious%':>9s}")
        for name, *_ in VARIANTS:
            st = stats[k][name]
            summ = st["cm"].summary(m.class_names)
            row = {"mean_dice": summ["mean_dice_present_fg"],
                   "per_class_dice": {c: r["dice"] for c, r in summ["per_class"].items() if r["present_in_ground_truth"]},
                   "pred_change_pct": 100 * st["pc"] / st["n"], "gt_change_pct": 100 * st["gc"] / st["n"],
                   "spurious_change_pct": 100 * st["sp"] / st["n"]}
            out["models"][k][name] = row
            print(f"{name:24s} {row['mean_dice']:8.4f} {row['pred_change_pct']:9.3f} {row['gt_change_pct']:8.3f} "
                  f"{row['spurious_change_pct']:9.3f}")
    ok = [c for c in cam_log if c["ok"]]
    out["camera"] = {"pairs": len(cam_log), "reliable_pairs": len(ok),
                     "median_translation_px": float(np.median([np.hypot(c["tx"], c["ty"]) for c in ok])) if ok else None,
                     "p90_translation_px": float(np.percentile([np.hypot(c["tx"], c["ty"]) for c in ok], 90)) if ok else None,
                     "median_abs_rotation_deg": float(np.median([abs(c["rot"]) for c in ok])) if ok else None,
                     "median_inlier_ratio": float(np.median([c["inlier_ratio"] for c in ok])) if ok else None,
                     "median_residual_px": float(np.median([c["residual_px"] for c in ok])) if ok else None,
                     "mean_ms": float(np.mean([c["ms"] for c in cam_log]))}
    print("\ncamera motion:", json.dumps(out["camera"], indent=1))
    od = resolve_path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"temporal_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {od / f'temporal_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
