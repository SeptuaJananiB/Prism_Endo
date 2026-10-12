"""Wider grasper test: human jaw-hinge clicks on 5 videos (17, 25, 28, 52, 55) vs the grasper methods.

Ground truth: prism-tips-GRASPER-annotator-* (human, jaw hinge, grasper_tip / grasper_tip_2), matched identity-free.
Methods (all use the single-video instrument model, trained on video01 only, so its masks never saw these videos):
  geometry_hinge : deployed default (mask tip, grasper jaw hinge 90 px back, fallback for short graspers)
  geometry_old   : previous rule (straight-line far end = jaw end)
  labeller_on_gt : the automatic labeller (auto_geo_v1) applied to the ground-truth CholecSeg8k masks - an upper bound
                   for perfect segmentation, not a deployable method
The grasper parameter (jaw_px = 90) was tuned on video01 training frames before these labels existed.
DLC is not evaluated for the grasper here: every DLC model trained on automatic grasper labels of these videos.

  python scripts/eval_grasper_wider.py
"""

import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
OUT = ROOT / "outputs/grasper_eval"


def gt_points():
    P = Path(sorted(glob.glob(str(ROOT / "outputs/dlc_projects/prism-tips-GRASPER-annotator-*")))[-1])
    rows = []
    for f in P.glob("labeled-data/*/CollectedData_annotator.h5"):
        clip = f.parent.name
        frames = sorted(int(p.name.split("_")[1]) for p in (ROOT / "data/cholecseg8k" / clip.split("_")[0] / clip)
                        .glob("frame_*_endo.png"))
        d = pd.read_hdf(f)
        for idx in d.index:
            img = idx[-1] if isinstance(idx, tuple) else Path(idx).name
            fn = int(img.split("_")[1])
            pts = [(float(d.loc[idx, ("annotator", b, "x")]), float(d.loc[idx, ("annotator", b, "y")]))
                   for b in ("grasper_tip", "grasper_tip_2")]
            rows.append({"clip": clip, "image": img, "frame_idx": fn,  # folder input: source_index = CholecSeg8k frame number
                         "pts": [p for p in pts if math.isfinite(p[0])]})
    return rows


def run(clip, method):
    out = OUT / method / clip
    if not (out / "results.csv").is_file():
        ov = ["instrument.checkpoint=outputs/models/instrument_seg/best.pt", "tissue.checkpoint=null",
              "keypoints.source=mask_tip", "safety.cvs.enabled=false", "proximity.approx_mm.enabled=false",
              "output.write_video=false"]
        if method == "geometry_scaled":
            ov.append("keypoints.mask_tip_landmarks={grasper: {landmark: hinge, jaw_rel: 2.0, step: 2}}")
        if method == "geometry_old":
            ov.append("keypoints.mask_tip_landmarks=null")
        v = clip.split("_")[0]
        subprocess.run([str(PY), "run_pipeline.py", "--input", f"data/cholecseg8k/{v}/{clip}", "--out", str(out), *ov],
                       cwd=ROOT, capture_output=True)
    return pd.read_csv(out / "results.csv").set_index("source_index")


def score(errs, n_gt, extra):
    e = np.array(errs)
    return {"gt_points": n_gt, "found": len(e), "missed": n_gt - len(e), "extra_points": extra,
            "median_px": float(np.median(e)) if len(e) else None, "mean_px": float(e.mean()) if len(e) else None,
            **{f"pck@{t}px": float((e <= t).mean()) if len(e) else None for t in (10, 20, 40)}}


def match(gts, preds):
    used, errs = set(), []
    for g in gts:
        c = [(math.dist(g, p), i) for i, p in enumerate(preds) if i not in used]
        if c:
            d, i = min(c)
            used.add(i)
            errs.append(d)
    return errs


def main():
    G = gt_points()
    auto = pd.read_csv(ROOT / "outputs/auto_labels/all_frames.csv")
    auto["name"] = auto["image"].map(lambda p: Path(p).name)
    ak = {(c, n): json.loads(g) if isinstance(g, str) else [] for c, n, g in zip(auto["clip"], auto["name"], auto["grasper"])}
    res, per_video = {}, {}
    for method in ("geometry_hinge", "geometry_scaled", "geometry_old", "labeller_on_gt"):
        errs, n_gt, extra = [], 0, 0
        vids = {}
        for r in G:
            if method == "labeller_on_gt":
                preds = [tuple(p[:2]) for p in ak.get((r["clip"], r["image"]), [])]
            else:
                df = run(r["clip"], method)
                row = df.loc[r["frame_idx"]] if r["frame_idx"] in df.index else None
                preds = []
                for s in ("grasper_tip", "grasper_tip_2"):
                    if row is not None and f"{s}_x" in row and str(row[f"{s}_kp_status"]) in ("ok", "low_confidence"):  # same rule as the video01 test (min-conf 0)
                        preds.append((float(row[f"{s}_x"]), float(row[f"{s}_y"])))
            e = match(r["pts"], preds)
            errs += e
            n_gt += len(r["pts"])
            extra += max(0, len(preds) - len(r["pts"]))
            v = r["clip"].split("_")[0]
            vids.setdefault(v, [[], 0])
            vids[v][0] += e
            vids[v][1] += len(r["pts"])
        res[method] = score(errs, n_gt, extra)
        per_video[method] = {v: score(x[0], x[1], 0) for v, x in sorted(vids.items())}
    report = {"ground_truth": "human jaw-hinge clicks, 40 frames, 5 videos", "frames": len(G),
              "overall": res, "per_video": per_video}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grasper_wider_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
