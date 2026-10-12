"""Prepare a HUMAN L-hook labelling set from videos other than video01 (for the click tool, src/ui/label_tips.py).

Selects frames whose ground-truth CholecSeg8k mask contains an L-hook, --per-clip evenly spaced frames from every
clip (all 6 non-video01 videos that contain the L-hook), copies them into a separate DLC-style project
outputs/dlc_projects/prism-tips-LHOOK-annotator-<date>/ (scorer 'annotator', same bodyparts as the other tip
projects so the labels can be merged later; only lhook_tip is labelled here).

  python scripts/prepare_lhook_labeling.py --per-clip 6
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-clip", type=int, default=6)
    a = ap.parse_args(argv)
    proj = ROOT / "outputs" / "dlc_projects" / f"prism-tips-LHOOK-annotator-{date.today():%Y-%m-%d}"
    if proj.exists():
        print(f"{proj} exists; not overwriting (your clicks may be in it)")
        return 1
    df = pd.read_csv(ROOT / "outputs" / "auto_labels" / "all_frames.csv")
    df = df[(df["video"] != "video01") & (df["lhook"].fillna("null") != "null")].copy()
    df["fnum"] = df["image"].str.extract(r"frame_(\d+)_endo")[0].astype(int)
    rows = []
    for clip, g in df.sort_values("fnum").groupby("clip"):
        idx = np.unique(np.linspace(0, len(g) - 1, min(a.per_clip, len(g))).round().astype(int))
        for r in g.iloc[idx].itertuples():
            gr = json.loads(r.grasper) if isinstance(r.grasper, str) else []
            rows.append({"clip": clip, "video": r.video, "image": Path(r.image).name, "cholecseg8k_frame": r.fnum,
                         "instruments_in_gt_mask": "l_hook" + (f"+{len(gr)}x_grasper" if gr else ""),
                         "source_path": r.image})
    man = pd.DataFrame(rows)
    for r in man.itertuples():
        out = proj / "labeled-data" / r.clip
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r.source_path, out / r.image)
    (proj / "videos").mkdir()
    base = yaml.safe_load(sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-eval-*"))[-1]
                          .joinpath("config.yaml").read_text(encoding="utf-8"))
    base.update(Task="prism-tips-LHOOK", scorer="annotator", project_path=str(proj),
                video_sets={str(proj / "videos" / f"{c}.mp4"): {"crop": "0, 854, 0, 480"} for c in man["clip"].unique()},
                label_only=["lhook_tip"])
    (proj / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    man.drop(columns=["source_path"]).to_csv(proj / "selection_manifest.csv", index=False)
    summary = {"frames": len(man), "clips": int(man["clip"].nunique()), "videos": sorted(man["video"].unique().tolist()),
               "per_video": man.groupby("video").size().to_dict(), "label": "lhook_tip only, human clicks",
               "excluded": "video01 (holds the 40 evaluation frames)"}
    (proj / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), "\nproject:", proj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
