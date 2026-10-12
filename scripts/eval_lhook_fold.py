"""Cross-video test: L-hook tip error on HELD-OUT patients (video52, video55) against your human clicks.

The DLC fold model never saw these two videos (prism-tips-V3fold-holdout5255-*). For each held-out clip the pipeline
runs with (a) the geometric mask-tip method and (b) the hybrid with the fold DLC model; both use the single-video
instrument model (trained on video01 only, so its masks have not seen these videos either). Errors are measured at
your labelled frames; "not visible" frames count as false alarms when a method reports a valid L-hook point.

  python scripts/eval_lhook_fold.py --dlc-dir outputs/dlc/v3fold_model
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def human_points():
    hum = Path(sorted(glob.glob(str(ROOT / "outputs/dlc_projects/prism-tips-LHOOK-annotator-*")))[-1])
    rows = []
    for f in hum.glob("labeled-data/video5[25]_*/CollectedData_annotator.h5"):
        clip = f.parent.name
        frames = sorted(int(p.name.split("_")[1]) for p in (ROOT / "data/cholecseg8k" / clip.split("_")[0] / clip)
                        .glob("frame_*_endo.png"))
        d = pd.read_hdf(f)
        for idx in d.index:
            img = idx[-1] if isinstance(idx, tuple) else Path(idx).name
            fn = int(img.split("_")[1])
            rows.append({"clip": clip, "frame_idx": frames.index(fn), "x": d.loc[idx, ("annotator", "lhook_tip", "x")],
                         "y": d.loc[idx, ("annotator", "lhook_tip", "y")]})
    return pd.DataFrame(rows)


def run(clip: str, out: Path, extra: list[str]) -> pd.DataFrame:
    if not (out / "results.csv").is_file():
        subprocess.run([str(PY), "run_pipeline.py", "--input", f"Videos/{clip}.mp4", "--out", str(out),
                        "instrument.checkpoint=outputs/models/instrument_seg/best.pt", "tissue.checkpoint=null",
                        *extra], cwd=ROOT, capture_output=True)
    return pd.read_csv(out / "results.csv").set_index("source_index")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dlc-dir", required=True, help="fold DLC predictions (<clip>DLC*.h5)")
    ap.add_argument("--out", default="outputs/fold_eval")
    a = ap.parse_args(argv)
    H = human_points()
    res = {}
    for method in ("geometry", "hybrid_lhook", "dlc_only"):
        errs, fa, n_vis, n_none, missed = [], 0, 0, 0, 0
        for clip, g in H.groupby("clip"):
            pred = [h for h in (ROOT / a.dlc_dir).glob(f"{clip}DLC*.h5") if "_meta" not in h.stem]
            if method == "geometry":
                df = run(clip, ROOT / a.out / method / clip, ["keypoints.source=mask_tip"])
            elif method == "hybrid_lhook":
                df = run(clip, ROOT / a.out / method / clip, ["keypoints.source=hybrid",
                         f"keypoints.dlc_predictions={pred[0].as_posix()}", "keypoints.hybrid={dlc_slots: [lhook_tip]}"])
            else:
                df = run(clip, ROOT / a.out / method / clip, ["keypoints.source=dlc", "keypoints.dlc_min_confidence=0.0",
                         f"keypoints.dlc_predictions={pred[0].as_posix()}"])
            for r in g.itertuples():
                row = df.loc[r.frame_idx] if r.frame_idx in df.index else None
                has = row is not None and str(row["lhook_tip_kp_status"]) == "ok" and np.isfinite(row["lhook_tip_x"])
                if np.isfinite(r.x):
                    n_vis += 1
                    if has:
                        errs.append(float(np.hypot(row["lhook_tip_x"] - r.x, row["lhook_tip_y"] - r.y)))
                    else:
                        missed += 1
                else:
                    n_none += 1
                    fa += bool(has)
        e = np.array(errs)
        res[method] = {"labelled_tips": n_vis, "found": len(e), "missed": missed,
                       "median_px": float(np.median(e)) if len(e) else None,
                       "mean_px": float(e.mean()) if len(e) else None,
                       "within_20px": float((e <= 20).mean()) if len(e) else None,
                       "not_visible_frames": n_none, "false_alarms_on_not_visible": fa}
    (ROOT / a.out).mkdir(parents=True, exist_ok=True)
    (ROOT / a.out / "fold_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
