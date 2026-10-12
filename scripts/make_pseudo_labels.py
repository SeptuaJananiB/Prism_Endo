"""Create a SEPARATE DeepLabCut project with MACHINE-GENERATED (pseudo) tip labels.

Pseudo labels = the mask-tip rule (src/tracking/mask_tip.py) applied to the GROUND-TRUTH CholecSeg8k instrument
segmentation of the 200 prepared training frames. They are NOT human keypoint labels:
  * written to a copy of the training project named prism-tips-PSEUDO-..., scorer 'pseudo_masktip';
  * the human projects (prism-tips-train/-eval) are never touched;
  * a model trained on them must not be reported as evaluated against human ground truth. Accuracy can only be
    measured on human-labelled frames (the 40 eval frames, once labelled).

  python scripts/make_pseudo_labels.py
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.tracking.mask_tip import estimate_tip, field_of_view_mask, fov_edge_distance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-train-*"))[-1]
DST = SRC.with_name(SRC.name.replace("prism-tips-train", "prism-tips-PSEUDO"))
SCORER = "pseudo_masktip"


def tips(img, lab):
    """Up to: lhook_tip (largest L-hook component), grasper_tip / grasper_tip_2 (two largest grasper
    components; _tip = component entering further LEFT, per the annotation guide)."""
    fov = field_of_view_mask(img)
    de = fov_edge_distance(fov)
    out = {}
    est = estimate_tip(lab == 32, fov, 300, 12, dist_edge=de)
    if est and est["confidence"] >= 0.3:
        out["lhook_tip"] = (est["x"], est["y"])
    n, cc, st, _ = cv2.connectedComponentsWithStats((lab == 31).astype(np.uint8), 8)
    comps = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]
    found = []
    for i in comps:
        if st[i, cv2.CC_STAT_AREA] < 300:
            continue
        e = estimate_tip(cc == i, fov, 300, 12, dist_edge=de)
        if e and e["confidence"] >= 0.3:
            found.append(e)
    found.sort(key=lambda e: e["entry"][0])  # entering from the left first
    for name, e in zip(("grasper_tip", "grasper_tip_2"), found):
        out[name] = (e["x"], e["y"])
    return out


def main():
    if DST.exists():
        print(f"{DST} already exists; not overwriting.")
        return 1
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("CollectedData_*", "reviewed.json", "dlc-models*",
                                                            "training-datasets"))
    cfg_path = DST / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["project_path"] = str(DST)
    cfg["scorer"] = SCORER
    cfg["video_sets"] = {str(DST / "videos" / Path(k).name): v for k, v in cfg["video_sets"].items()}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    bps = cfg["bodyparts"]
    man = pd.read_csv(DST / "selection_manifest.csv")
    counts = {b: 0 for b in bps}
    for clip, g in man.groupby("clip"):
        rows, idx = [], []
        for r in g.itertuples():
            img = cv2.imread(str(DST / "labeled-data" / clip / r.image))
            m = ROOT / "video01" / clip / f"frame_{r.cholecseg8k_frame}_endo_watershed_mask.png"
            lab = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            t = tips(img, lab)
            for b in t:
                counts[b] += 1
            idx.append(("labeled-data", clip, r.image))
            rows.append([v for b in bps for v in t.get(b, (np.nan, np.nan))])
        cols = pd.MultiIndex.from_product([[SCORER], bps, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
        df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx), columns=cols).astype(float)
        df.to_hdf(DST / "labeled-data" / clip / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
        df.to_csv(DST / "labeled-data" / clip / f"CollectedData_{SCORER}.csv")
    note = {"type": "PSEUDO LABELS - machine generated, not human ground truth",
            "method": "mask-tip rule on ground-truth CholecSeg8k instrument masks", "points_per_bodypart": counts,
            "frames": int(len(man))}
    (DST / "PSEUDO_LABELS_README.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2), "\nproject:", cfg_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
