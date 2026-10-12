"""Score a TipNet checkpoint on the 40 HUMAN-labelled evaluation frames (video01 clips, never used in training).

Same protocol as evaluate_dlc_heldout.py: flagged labels in outputs/label_qa/excluded.json are skipped, graspers are
matched identity-free (nearest unused prediction). The decoding threshold is fixed (default 0.3) - NOT tuned on these
frames. Also reports CPU inference time per frame.

  .venv\\Scripts\\python scripts\\eval_tipnet_human.py --ckpt outputs/models/tipnet/best.pt
"""

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.keypoints import summarize_errors  # noqa: E402
from src.tracking import tipnet as tn  # noqa: E402
from src.tracking.dlc_io import read_dlc_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="outputs/models/tipnet/best.pt")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/models/tipnet/human_eval.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    model = tn.build_model(pretrained=False)
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state"])
    model.eval()
    E = Path(sorted(glob.glob(str(ROOT / "outputs/dlc_projects/prism-tips-eval-*")))[-1])
    man = pd.read_csv(E / "selection_manifest.csv")
    skip = {(e["clip"], e["image"], e["bodypart"]) for e in json.loads((ROOT / "outputs/label_qa/excluded.json").read_text())}
    res = {"lhook": {"err": [], "gt": 0, "extra": 0}, "grasper": {"err": [], "gt": 0, "extra": 0}}
    times, per_frame = [], []
    for clip, g in man.groupby("clip"):
        df = read_dlc_table(glob.glob(str(E / "labeled-data" / clip / "CollectedData_*.h5"))[0])
        for r in g.itertuples():
            row = df.loc[[i for i in df.index if str(i[-1]) == r.image][0]]

            def pt(bp):
                v = row.xs(bp, level="bodyparts")
                return float(v.xs("x", level="coords").iloc[0]), float(v.xs("y", level="coords").iloc[0])

            img = cv2.imread(str(E / "labeled-data" / clip / r.image))
            t0 = time.perf_counter()
            with torch.no_grad():
                heat = torch.sigmoid(model(torch.from_numpy(tn.preprocess(img))[None]))[0].numpy()
            d = tn.decode(heat, a.thr)
            times.append(time.perf_counter() - t0)
            for grp, bps, preds in (("lhook", ["lhook_tip"], d["lhook_tip"]),
                                    ("grasper", ["grasper_tip", "grasper_tip_2"], d["grasper_hinge"])):
                gts = [pt(b) for b in bps if all(map(math.isfinite, pt(b))) and (clip, r.image, b) not in skip]
                res[grp]["gt"] += len(gts)
                used = set()
                for gp in gts:
                    c = [(math.dist(gp, p[:2]), i) for i, p in enumerate(preds) if i not in used]
                    if c:
                        e, i = min(c)
                        used.add(i)
                        res[grp]["err"].append(e)
                        per_frame.append({"clip": clip, "image": r.image, "group": grp, "err_px": round(e, 1)})
                res[grp]["extra"] += max(0, len(preds) - len(gts))
    report = {"ckpt": a.ckpt, "epoch": ck.get("epoch"), "thr": a.thr, "cpu_ms_per_frame": round(1000 * np.median(times[1:]), 1)}
    for grp, v in res.items():
        e = np.array(v["err"])
        report[grp] = {"gt_points": v["gt"], "matched": len(e), "detection_recall": len(e) / max(1, v["gt"]),
                       "extra_preds": v["extra"], **summarize_errors(e, image_diag=math.hypot(854, 480))}
    Path(ROOT / a.out).write_text(json.dumps({**report, "per_point": per_frame}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
