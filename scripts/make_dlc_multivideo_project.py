"""Build a DeepLabCut project from the automatic PSEUDO tip labels of 16 CholecSeg8k videos (video01 excluded).

Source: outputs/auto_labels/all_frames.csv (scripts/auto_label_all.py, method auto_geo_v1 on ground-truth masks).
Every --stride-th frame of each clip is used (consecutive frames are near-duplicates). Bodyparts match the existing
tip projects: lhook_tip, grasper_tip (left-most hinge), grasper_tip_2. Images are copied into labeled-data/<clip>/.
These are machine labels, NOT human ground truth; the model is scored only on the 40 human eval frames (video01).

  python scripts/make_dlc_multivideo_project.py --stride 6
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCORER = "auto_geo_v1"
BPS = ["lhook_tip", "grasper_tip", "grasper_tip_2"]


def _j(v):
    return json.loads(v) if isinstance(v, str) and v else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=6, help="every Nth frame of each clip (all frames)")
    ap.add_argument("--lhook-stride", type=int, default=0,
                    help="if >0: frames WITH an L-hook use this stride, frames without it use --stride (balances the rare L-hook)")
    ap.add_argument("--name", default="prism-tips-MULTI16-auto-2026-10-10")
    ap.add_argument("--exclude-videos", nargs="*", default=[], help="videos left out entirely (clean held-out tests)")
    a = ap.parse_args(argv)
    dst = ROOT / "outputs" / "dlc_projects" / a.name
    if dst.exists():
        print(f"{dst} exists; not overwriting")
        return 1
    src_cfg = sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-train-*"))[-1] / "config.yaml"
    cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / "outputs" / "auto_labels" / "all_frames.csv")
    assert not (df.video == "video01").any()
    df = df[~df["video"].isin(a.exclude_videos)]
    df["fnum"] = df.image.str.extract(r"frame_(\d+)_endo")[0].astype(int)
    rows = []
    for clip, g in df.sort_values("fnum").groupby("clip"):
        if a.lhook_stride:
            has_l = g.lhook.map(lambda v: _j(v) is not None)
            rows += [g[has_l].iloc[:: a.lhook_stride], g[~has_l].iloc[:: a.stride]]
        else:
            rows.append(g.iloc[:: a.stride])
    sel = pd.concat(rows)
    (dst / "labeled-data").mkdir(parents=True)
    (dst / "videos").mkdir()
    cols = pd.MultiIndex.from_product([[SCORER], BPS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    video_sets, counts = {}, {b: 0 for b in BPS}
    for clip, g in sel.groupby("clip"):
        out = dst / "labeled-data" / clip
        out.mkdir()
        recs, idx = [], []
        for r in g.itertuples():
            name = Path(r.image).name
            shutil.copy2(ROOT / r.image, out / name)
            lh = _j(r.lhook)
            gr = sorted(_j(r.grasper) or [], key=lambda p: p[0])[:2]  # left-most hinge first
            pts = {"lhook_tip": lh[:2] if lh else None,
                   "grasper_tip": gr[0][:2] if len(gr) > 0 else None,
                   "grasper_tip_2": gr[1][:2] if len(gr) > 1 else None}
            for b in BPS:
                counts[b] += pts[b] is not None
            recs.append([v for b in BPS for v in (pts[b] if pts[b] else (np.nan, np.nan))])
            idx.append(("labeled-data", clip, name))
        t = pd.DataFrame(recs, index=pd.MultiIndex.from_tuples(idx), columns=cols).astype(float)
        t.to_hdf(out / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
        t.to_csv(out / f"CollectedData_{SCORER}.csv")
        video_sets[str(dst / "videos" / f"{clip}.mp4")] = {"crop": "0, 854, 0, 480"}
    cfg.update(Task=a.name.rsplit("-", 4)[0], scorer=SCORER, project_path=str(dst), video_sets=video_sets,
               bodyparts=BPS)
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    note = {"type": "PSEUDO LABELS (auto_geo_v1 on GT masks) - not human ground truth", "stride": a.stride,
            "lhook_stride": a.lhook_stride,
            "frames": int(len(sel)), "clips": int(sel["clip"].nunique()), "videos": sorted(sel.video.unique().tolist()),
            "points_per_bodypart": counts, "excluded": ["video01 (holds the 40 human eval frames)"] + list(a.exclude_videos)}
    (dst / "PSEUDO_LABELS_README.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2), "\nproject:", dst / "config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
