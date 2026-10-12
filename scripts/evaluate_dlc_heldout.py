"""Evaluate tip predictions on the HELD-OUT eval project's human labels.

Predictions come either from DLC (`<video>DLC_*.h5` files under --pred-dir, from
`dlc_workflow.py analyze`) or from PRISM-Endo pipeline runs (--pipeline-runs, using
outputs/runs/<video>/results.csv; e.g. the mask-tip heuristic baseline).
Grasper tips are scored identity-free (each predicted grasper point is matched to the
nearest labelled grasper) because which grasper is "first" is a labelling convention.
Prints "no labels" instead of numbers when nothing is labelled yet.

  python scripts/evaluate_dlc_heldout.py --eval-config <eval project>/config.yaml --pipeline-runs outputs/runs
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.keypoints import summarize_errors  # noqa: E402
from src.tracking.dlc_io import DLCPredictions  # noqa: E402
from src.tracking.dlc_workflow import label_status  # noqa: E402

GROUPS = {"lhook": ["lhook_tip"], "grasper": ["grasper_tip", "grasper_tip_2"]}


def _pred_points(src, frame, names, min_conf):
    pts = []
    for n in names:
        r = src(frame, n)
        if r is not None and all(map(math.isfinite, r[:2])) and r[2] >= min_conf:
            pts.append(r[:2])
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--pred-dir", help="folder tree containing <video>DLC_*.h5")
    ap.add_argument("--pipeline-runs", help="folder with <video>/results.csv")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--exclude", default=None, help="JSON list of {clip, image, bodypart} labels to skip "
                    "(flagged for human review; labels themselves are not modified)")
    args = ap.parse_args(argv)
    cfg = Path(args.eval_config)
    st = label_status(cfg)
    if st["frames_labelled"] == 0:
        print(f"No human labels found in {cfg.parent / 'labeled-data'} - cannot evaluate. "
              f"Label the eval project first (docs/DLC_ANNOTATION.md).")
        return 2
    skip = set()
    if args.exclude:
        skip = {(e["clip"], e["image"], e["bodypart"]) for e in json.loads(Path(args.exclude).read_text())}
    errors = {g: [] for g in GROUPS}
    counts = {g: {"gt_points": 0, "matched": 0, "confident_preds_without_gt": 0} for g in GROUPS}
    for vid_dir in sorted((cfg.parent / "labeled-data").iterdir()):
        files = sorted(vid_dir.glob("CollectedData_*.h5")) or sorted(vid_dir.glob("CollectedData_*.csv"))
        if not files:
            continue
        gt = DLCPredictions(files[0])
        from src.tracking.dlc_io import frame_index_from_row_labels, read_dlc_table
        gt_index = list(read_dlc_table(files[0]).index)
        gt_frame = lambda i: int(frame_index_from_row_labels(pd.Index([i]))[0])  # noqa: E731
        if args.pred_dir:
            hits = [h for h in Path(args.pred_dir).rglob(f"{vid_dir.name}DLC*.h5") if "_meta" not in h.stem]
            if not hits:
                print(f"no DLC prediction for {vid_dir.name}; skipped")
                continue
            dp = DLCPredictions(hits[0])
            src = lambda f, n: dp.get(f, n) if n in dp.bodyparts else None  # noqa: E731
        else:
            csv = Path(args.pipeline_runs) / vid_dir.name / "results.csv"
            if not csv.is_file():
                print(f"no pipeline results for {vid_dir.name}; skipped")
                continue
            df = pd.read_csv(csv).set_index("source_index")

            def src(f, n, df=df):
                if f not in df.index or f"{n}_x" not in df.columns:
                    return None
                r = df.loc[f]
                ok = str(r[f"{n}_kp_status"]) == "ok"
                return float(r[f"{n}_x"]), float(r[f"{n}_y"]), float(r[f"{n}_confidence"]) if ok else 0.0
        for f in sorted(int(x) for x in gt.frames if x >= 0):
            for g, names in GROUPS.items():
                img_name = next((str(i[-1]) for i in gt_index if gt_frame(i) == f), "")
                gts = [gt.get(f, n)[:2] for n in names if n in gt.bodyparts and gt.get(f, n)
                       and all(map(math.isfinite, gt.get(f, n)[:2])) and (vid_dir.name, img_name, n) not in skip]
                preds = _pred_points(src, f, names, args.min_conf)
                counts[g]["gt_points"] += len(gts)
                used = set()
                for gp in gts:  # nearest unused prediction
                    if not preds:
                        break
                    d = [math.hypot(p[0] - gp[0], p[1] - gp[1]) if i not in used else 1e9 for i, p in enumerate(preds)]
                    i = int(np.argmin(d))
                    if d[i] < 1e9:
                        used.add(i)
                        errors[g].append(d[i])
                counts[g]["confident_preds_without_gt"] += max(0, len(preds) - len(gts))
    diag = math.hypot(args.width, args.height)
    res = {g: {**counts[g], "matched": len(errors[g]),
               "detection_recall": len(errors[g]) / counts[g]["gt_points"] if counts[g]["gt_points"] else None,
               **summarize_errors(np.array(errors[g]), image_diag=diag)} for g in GROUPS}
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
