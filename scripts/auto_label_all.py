"""Automatic (PSEUDO) tip labels for every CholecSeg8k frame of every video EXCEPT video01.

video01 contains the 40 human-labelled evaluation frames, so it is excluded entirely (no near-duplicate leakage).
Method auto_geo_v1 (src/tracking/auto_tips.py) on the GROUND-TRUTH instrument masks:
  lhook  = geodesic far end of the largest L-hook component;
  grasper = jaw hinge (90 px back from the far end) for up to 2 components; a component whose visible length is
            < jaw_px + 10 has no locatable hinge -> no point (never a far-end substitute in TRAINING labels).
Output: outputs/auto_labels/all_frames.csv (one row per frame; empty = instrument absent).

  python scripts/auto_label_all.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from src.tracking.auto_tips import components, tip_from_component  # noqa: E402
from src.tracking.mask_tip import field_of_view_mask, fov_edge_distance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cholecseg8k"
OUT = ROOT / "outputs" / "auto_labels" / "all_frames.csv"
JAW = 90


def label_frame(img_path: Path) -> dict:
    lab = cv2.imread(str(img_path).replace("_endo.png", "_endo_watershed_mask.png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(str(img_path))
    de = fov_edge_distance(field_of_view_mask(img))
    row = {}
    lh = components(lab == 32, keep=1)
    t = tip_from_component(lh[0], de, "far_end", step=2) if lh else None
    row["lhook"] = [round(t["x"], 1), round(t["y"], 1), round(t["confidence"], 3)] if t else None
    gr = []
    for c in components(lab == 31, keep=2):
        t = tip_from_component(c, de, "hinge", jaw_px=JAW, step=2)
        if t and t["geo_len"] >= JAW + 10:
            gr.append([round(t["x"], 1), round(t["y"], 1), round(t["confidence"], 3)])
    row["grasper"] = gr
    return row


def main():
    rows = []
    vids = sorted(p for p in DATA.glob("video*") if p.is_dir() and p.name != "video01")
    for v in vids:
        for img in sorted(v.glob("*/*_endo.png")):
            r = label_frame(img)
            rows.append({"video": v.name, "clip": img.parent.name, "image": str(img.relative_to(ROOT)),
                         "lhook": json.dumps(r["lhook"]), "grasper": json.dumps(r["grasper"])})
        print(v.name, len(rows), flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    n_l = (df.lhook != "null").sum()
    n_g = df.grasper.map(lambda s: len(json.loads(s))).sum()
    print(f"frames {len(df)} from {df.video.nunique()} videos; lhook points {n_l}; grasper hinge points {n_g}")


if __name__ == "__main__":
    main()
