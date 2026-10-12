"""Automatic (PSEUDO) tip labels for the 200 prepared training frames - no manual clicking.

Method `auto_geo_v1` (src/tracking/auto_tips.py) on the GROUND-TRUTH CholecSeg8k instrument masks of each frame:
  * lhook_tip     : geodesic far end of the L-hook mask from its field-of-view entry;
  * grasper_tip(_2): jaw HINGE = 90 px back from the geodesic far end along the instrument (jaw_px=90, tuned on
                    training frames only; the 40 human eval frames were NOT used for tuning).
Points are only given when the instrument is in the GT mask (absent -> missing). Each point records confidence and
its disagreement with the old mask-tip rule, plus a temporal check against the neighbouring selected frame of the same clip. Flags mark points for review.

Outputs (outputs/auto_labels/): train_auto_labels.csv, summary.json, contact_sheet_*.jpg, and a separate DLC project
prism-tips-AUTO-* (scorer auto_geo_v1). These are machine labels, NOT human-verified ground truth.

  python scripts/auto_label_tips.py
"""

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from make_pseudo_labels import tips as old_masktip  # noqa: E402
from src.tracking.auto_tips import auto_tips  # noqa: E402
from src.tracking.mask_tip import field_of_view_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-train-*"))[-1]
OUT = ROOT / "outputs" / "auto_labels"
METHOD, JAW_PX = "auto_geo_v1", 90
BPS = ("lhook_tip", "grasper_tip", "grasper_tip_2")
FLAG_JUMP_PX, FLAG_CONF = 80.0, 0.3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(SRC / "selection_manifest.csv")
    rows = []
    for r in man.itertuples():
        img = cv2.imread(str(SRC / "labeled-data" / r.clip / r.image))
        lab = cv2.imread(str(ROOT / "video01" / r.clip / f"frame_{r.cholecseg8k_frame}_endo_watershed_mask.png"),
                         cv2.IMREAD_GRAYSCALE)
        t = auto_tips(lab, field_of_view_mask(img), "hinge", jaw_px=JAW_PX)
        old = old_masktip(img, lab)
        for b in BPS:
            p = t[b]
            rec = {"clip": r.clip, "image": r.image, "video": "video01", "cholecseg8k_frame": r.cholecseg8k_frame,
                   "video_frame_index": r.video_frame_index, "bodypart": b, "visible": p is not None,
                   "x": round(p["x"], 1) if p else np.nan, "y": round(p["y"], 1) if p else np.nan,
                   "confidence": round(p["confidence"], 3) if p else 0.0, "method": METHOD if p else "absent_in_gt_mask",
                   "geo_len_px": p["geo_len"] if p else np.nan, "shaft_width_px": p["shaft_width"] if p else np.nan}
            # disagreement with the previous rule (L-hook only)
            if b == "lhook_tip" and p and "lhook_tip" in old:
                rec["old_masktip_dist_px"] = round(math.dist((p["x"], p["y"]), old["lhook_tip"]), 1)
            else:
                rec["old_masktip_dist_px"] = np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    # temporal consistency: jump to the previous selected frame of the same clip (same bodypart)
    df["temporal_jump_px"] = np.nan
    for (clip, b), g in df.groupby(["clip", "bodypart"]):
        g = g.sort_values("video_frame_index")
        prev = None
        for i, rr in g.iterrows():
            if rr.visible and prev is not None and prev.visible:
                df.loc[i, "temporal_jump_px"] = round(math.hypot(rr.x - prev.x, rr.y - prev.y), 1)
            prev = rr
    flags = []
    for rr in df.itertuples():
        f = []
        if rr.visible and rr.confidence < FLAG_CONF:
            f.append("low_confidence")
        if rr.temporal_jump_px == rr.temporal_jump_px and rr.temporal_jump_px > FLAG_JUMP_PX:
            f.append("temporal_jump")
        flags.append(";".join(f))
    df["flags"] = flags
    df.to_csv(OUT / "train_auto_labels.csv", index=False)

    vis = df[df.visible]
    summ = {"type": "PSEUDO LABELS (machine-generated, not human-verified)", "method": METHOD, "jaw_px": JAW_PX,
            "grasper_definition": "jaw hinge", "frames": int(man.shape[0]),
            "points": {b: int(vis[vis.bodypart == b].shape[0]) for b in BPS},
            "flagged_points": int((vis["flags"] != "").sum()),
            "flag_counts": vis["flags"].str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict(),
            "median_confidence": {b: float(vis[vis.bodypart == b].confidence.median()) for b in BPS}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    contact_sheets(df)
    write_dlc_project(df)
    print(json.dumps(summ, indent=2))


def contact_sheets(df, per=40, cols=8, w=214, h=120):
    colour = {"lhook_tip": (255, 0, 255), "grasper_tip": (0, 255, 255), "grasper_tip_2": (0, 165, 255)}
    frames = df.drop_duplicates(["clip", "image"])[["clip", "image"]].values.tolist()
    for s in range(0, len(frames), per):
        tiles = []
        for clip, image in frames[s:s + per]:
            im = cv2.imread(str(SRC / "labeled-data" / clip / image))
            g = df[(df["clip"] == clip) & (df["image"] == image)]
            for rr in g[g.visible].itertuples():
                c = colour[rr.bodypart]
                cv2.circle(im, (int(rr.x), int(rr.y)), 18, c, -1)
                if rr.flags:
                    cv2.circle(im, (int(rr.x), int(rr.y)), 30, (0, 0, 255), 6)
            cv2.putText(im, f"{clip[-5:]}/{image}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            tiles.append(cv2.resize(im, (w, h)))
        while len(tiles) % cols:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(OUT / f"contact_sheet_{s // per + 1}.jpg"), sheet)


def write_dlc_project(df):
    dst = SRC.with_name(SRC.name.replace("prism-tips-train", "prism-tips-AUTO"))
    if dst.exists():
        print(f"{dst} exists; DLC project not rewritten")
        return
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("CollectedData_*", "reviewed.json", "dlc-models*",
                                                            "training-datasets"))
    cfg_p = dst / "config.yaml"
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    cfg.update(project_path=str(dst), scorer=METHOD)
    cfg["video_sets"] = {str(dst / "videos" / Path(k).name): v for k, v in cfg["video_sets"].items()}
    cfg_p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    bps = cfg["bodyparts"]
    cols = pd.MultiIndex.from_product([[METHOD], bps, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    for clip, g in df.groupby("clip"):
        recs, ix = [], []
        for image, gg in g.groupby("image", sort=False):
            v = gg.set_index("bodypart")
            recs.append([v.loc[b, c] if v.loc[b, "visible"] else np.nan for b in bps for c in ("x", "y")])
            ix.append(("labeled-data", clip, image))
        out = pd.DataFrame(recs, index=pd.MultiIndex.from_tuples(ix), columns=cols).astype(float)
        out.to_hdf(dst / "labeled-data" / clip / f"CollectedData_{METHOD}.h5", key="df_with_missing", mode="w")
        out.to_csv(dst / "labeled-data" / clip / f"CollectedData_{METHOD}.csv")
    (dst / "PSEUDO_LABELS_README.json").write_text(json.dumps(
        {"type": "PSEUDO LABELS - auto_geo_v1 on GT masks, not human ground truth", "grasper": "jaw hinge"}, indent=2))
    print("DLC project:", cfg_p)


if __name__ == "__main__":
    main()
