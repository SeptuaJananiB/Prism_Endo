"""Evaluate the EndoVis robotic DLC model on the OFFICIAL EndoVis'15 test sequences (never seen in training).

Test videos: Tracking_Robotic_Testing.zip (public KIT archive mirror), labels: EndoVisPoseAnnotation test_labels
(Du et al. 2018). Only labelled frames are analysed: they are copied losslessly (FFV1) into one short clip per
sequence, so frame k of the clip = k-th labelled frame.

Scoring is IDENTITY-FREE per joint type: tool ids differ between training (tool1/tool2) and test sequences 5-6
(tool1/tool4, some EndPoints without id), so each labelled joint is matched to the nearest unused confident prediction
of the same joint type from either predicted tool. Sequences 1-4 are 15 s clips from the training recording days;
5-6 are new 1-min sequences with two instruments and occlusions.

  python scripts/endovis_evaluate_test.py --config outputs/dlc_projects/endovis-robotic-du2018-2026-10-10/config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.keypoints import summarize_errors  # noqa: E402
from src.external.endovis import JOINTS, load_labels, read_frame  # noqa: E402
from src.tracking import dlc_workflow as wf  # noqa: E402
from src.tracking.dlc_io import DLCPredictions  # noqa: E402

DL = Path.home() / "Downloads"
ANN = DL / "PRISM_Endo_Public_Keypoint_Download_Kit/PRISM_Endo_ExternalData/EndoVisPoseAnnotation/EndoVisPoseAnnotation-master"
TEST = DL / "PRISM_Endo_ExternalData_Tracking/Tracking_Robotic_Testing"
PRED_TOOLS = ("tool1", "tool2")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pcutoff", type=float, default=0.6)
    ap.add_argument("--out", default="outputs/endovis/eval_official_test")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for js in sorted((ANN / "test_labels").glob("*.json")):
        frames += load_labels(js, "test")[0]
    per_seq, per_joint = {}, {j: {"err": [], "labelled": 0} for j in JOINTS}
    for seq in sorted({f.seq for f in frames}):
        lab = sorted((f for f in frames if f.seq == seq and f.points), key=lambda f: f.frame_number)
        clip = out / f"endovis_test_seq{seq}_labelled.avi"
        if not clip.exists():
            vw = None
            for f in lab:
                img = read_frame(TEST / f"Dataset{seq}" / "Video.avi", f.frame_number)
                if vw is None:
                    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"FFV1"), 25, (img.shape[1], img.shape[0]))
                vw.write(img)
            vw.release()
        t0 = time.perf_counter()
        pred = DLCPredictions(wf.analyze(a.config, clip, out, a.device, labeled_video=False))
        secs = time.perf_counter() - t0
        errs = []
        for k, f in enumerate(lab):
            for j in JOINTS:
                gts = [xy for (tool, jj), xy in f.points.items() if jj == j]
                preds = [p for p in (pred.get(k, f"{t}_{j}") for t in PRED_TOOLS)
                         if p is not None and np.isfinite(p[0]) and p[2] >= a.pcutoff]
                used = set()
                per_joint[j]["labelled"] += len(gts)
                for g in gts:
                    c = [(float(np.hypot(p[0] - g[0], p[1] - g[1])), i) for i, p in enumerate(preds) if i not in used]
                    if c:
                        d, i = min(c)
                        used.add(i)
                        per_joint[j]["err"].append(d)
                        errs.append(d)
        n_lab = sum(len(f.points) for f in lab)
        per_seq[f"Dataset{seq}"] = {"labelled_frames": len(lab), "labelled_joints": n_lab, "detected": len(errs),
                                    "detection_recall": len(errs) / n_lab if n_lab else None,
                                    "analyze_fps": round(len(lab) / secs, 2),
                                    **summarize_errors(np.array(errs), thresholds_px=(5, 10, 20))}
    allerr = np.concatenate([np.array(v["err"]) for v in per_joint.values()])
    n_all = sum(v["labelled"] for v in per_joint.values())
    report = {"split": "official EndoVis'15 robotic tracking TEST (Dataset1-6)", "pcutoff": a.pcutoff,
              "matching": "identity-free per joint type",
              "overall": {"labelled_joints": n_all, "detected": int(len(allerr)), "detection_recall": len(allerr) / n_all,
                          **summarize_errors(allerr, thresholds_px=(5, 10, 20))},
              "per_sequence": per_seq,
              "per_joint": {j: {"labelled": v["labelled"], "detected": len(v["err"]),
                                **summarize_errors(np.array(v["err"]), thresholds_px=(5, 10, 20))}
                            for j, v in per_joint.items()},
              "note": "Robotic ex-vivo instruments; not comparable to the laparoscopic L-hook/grasper tips."}
    (out / "endovis_official_test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("overall", "per_sequence")}, indent=2))


if __name__ == "__main__":
    main()
