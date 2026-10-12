"""Evaluate a trained EndoVis DLC model on HELD-OUT sequences against the human joint labels.

Runs deeplabcut.analyze_videos on each held-out Dataset video (never in the DLC project), then
compares predictions with the JSON labels: per joint, per tool, mean / median pixel error,
PCK@5/10/20 px, detection recall (prediction with likelihood >= --pcutoff where a label exists),
confident predictions where the joint was not labelled, and measured inference speed.

  python scripts/endovis_evaluate.py --config outputs/dlc_projects/endovis-robotic-du2018-2026-10-10/config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.endovis_validate import ANN, TRAIN  # noqa: E402
from src.evaluation.keypoints import summarize_errors  # noqa: E402
from src.external.endovis import JOINTS, load_labels  # noqa: E402
from src.tracking import dlc_workflow as wf  # noqa: E402
from src.tracking.dlc_io import DLCPredictions  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--heldout-seqs", type=int, nargs="+", default=[4])
    ap.add_argument("--pcutoff", type=float, default=0.6)
    ap.add_argument("--out", default="outputs/endovis/eval")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for js in sorted((ANN / "train_labels").glob("*.json")):
        frames += load_labels(js, "train")[0]
    results, errors = {}, {}
    for s in a.heldout_seqs:
        video = out / f"endovis_seq{s}_heldout.avi"
        if not video.exists():
            video.write_bytes((TRAIN / f"Dataset{s}" / "Video.avi").read_bytes())
        t0 = time.perf_counter()
        pred_file = wf.analyze(a.config, video, out, a.device, labeled_video=False)
        secs = time.perf_counter() - t0
        pred = DLCPredictions(pred_file)
        gt = [f for f in frames if f.seq == s and f.points and not any(i.startswith("duplicate") for i in f.issues)]
        for f in gt:
            for tool in ("tool1", "tool2"):
                for j in JOINTS:
                    bp = f"{tool}_{j}"
                    e = errors.setdefault(bp, {"err": [], "labelled": 0, "detected": 0, "conf_unlabelled": 0})
                    p = pred.get(f.frame_number - 1, bp)
                    confident = p is not None and np.isfinite(p[0]) and p[2] >= a.pcutoff
                    if (tool, j) in f.points:
                        e["labelled"] += 1
                        if confident:
                            e["detected"] += 1
                            gx, gy = f.points[(tool, j)]
                            e["err"].append(float(np.hypot(p[0] - gx, p[1] - gy)))
                    elif confident:
                        e["conf_unlabelled"] += 1
        n_video = len(pred.frames)
        results[f"Dataset{s}"] = {"labelled_frames": len(gt), "video_frames": n_video,
                                  "analyze_seconds": round(secs, 1), "analyze_fps": round(n_video / secs, 2)}
    per_bp = {}
    for bp, e in errors.items():
        s = summarize_errors(np.array(e["err"]), thresholds_px=(5, 10, 20))
        per_bp[bp] = {"labelled": e["labelled"], "detection_recall": e["detected"] / e["labelled"] if e["labelled"] else None,
                      "confident_where_unlabelled": e["conf_unlabelled"], **s}
    allerr = np.concatenate([np.array(e["err"]) for e in errors.values()]) if errors else np.array([])
    report = {"heldout": results, "pcutoff": a.pcutoff, "overall": summarize_errors(allerr, thresholds_px=(5, 10, 20)),
              "per_bodypart": per_bp,
              "note": "Robotic EndoVis'15 instruments (ex vivo). Not comparable to, and not used for, the laparoscopic "
                      "PRISM-Endo L-hook/grasper model."}
    (out / "endovis_heldout_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"heldout": results, "overall": report["overall"]}, indent=2))
    print(f"{'bodypart':28s} {'n':>4s} {'recall':>6s} {'mean':>6s} {'median':>6s} {'pck10':>6s}")
    for bp, r in per_bp.items():
        f = lambda v: f"{v:6.2f}" if isinstance(v, float) else "   n/a"  # noqa: E731
        print(f"{bp:28s} {r['labelled']:4d} {f(r['detection_recall'])} {f(r['mean_px'])} {f(r['median_px'])} {f(r.get('pck@10px'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
