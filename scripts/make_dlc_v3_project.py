"""DLC v3 training project: HUMAN L-hook clicks + automatic grasper labels (16 videos, video01 excluded).

Why: the CholecSeg8k L-hook masks often omit the thin metal hook, so automatic L-hook labels (mask far end) stop
20-50 px short of the true tip in those frames (outputs/label_qa/lhook_offmask_cases.jpg). v3 therefore:
  * takes the L-hook ONLY from human clicks (prism-tips-LHOOK-annotator-*; "not visible" frames kept as negatives);
  * drops every v2 frame that has an AUTOMATIC L-hook label (an unlabelled visible L-hook would be trained as
    "no tip here", so those frames cannot simply have the L-hook removed);
  * keeps the automatic grasper hinge labels (v2 frames without an L-hook + graspers of the human frames).

  python scripts/make_dlc_v3_project.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "outputs" / "dlc_projects"
SCORER = "human_lhook_auto_grasper"
BPS = ["lhook_tip", "grasper_tip", "grasper_tip_2"]


def main():
    v2 = sorted(PROJ.glob("prism-tips-MULTI16v2-auto-*"))[-1]
    hum = sorted(PROJ.glob("prism-tips-LHOOK-annotator-*"))[-1]
    dst = PROJ / "prism-tips-V3humanlhook-mixed-2026-10-11"
    if dst.exists():
        print(f"{dst} exists; not overwriting")
        return 1
    auto = pd.read_csv(ROOT / "outputs" / "auto_labels" / "all_frames.csv")
    auto["name"] = auto["image"].map(lambda p: Path(p).name)
    graspers = {(c, n): (json.loads(g) if isinstance(g, str) else []) for c, n, g in zip(auto["clip"], auto["name"], auto["grasper"])}
    rows = {}  # (clip, image) -> {"src": path, "pts": {bp: (x, y)}}
    # 1) v2 frames WITHOUT an automatic L-hook label (grasper-only frames)
    for f in v2.glob("labeled-data/*/CollectedData_*.h5"):
        d = pd.read_hdf(f)
        sc = d.columns.get_level_values(0)[0]
        for idx in d.index:
            img = idx[-1] if isinstance(idx, tuple) else Path(idx).name
            if np.isfinite(d.loc[idx, (sc, "lhook_tip", "x")]):
                continue
            rows[(f.parent.name, img)] = {"src": f.parent / img, "pts": {
                b: (d.loc[idx, (sc, b, "x")], d.loc[idx, (sc, b, "y")]) for b in BPS}}
    n_v2 = len(rows)
    # 2) human L-hook frames (clicked or "not visible"), graspers from the automatic labels
    n_h = n_vis = 0
    for f in hum.glob("labeled-data/*/CollectedData_annotator.h5"):
        d = pd.read_hdf(f)
        for idx in d.index:
            img = idx[-1] if isinstance(idx, tuple) else Path(idx).name
            x, y = d.loc[idx, ("annotator", "lhook_tip", "x")], d.loc[idx, ("annotator", "lhook_tip", "y")]
            gr = sorted(graspers.get((f.parent.name, img), []), key=lambda p: p[0])[:2]
            pts = {"lhook_tip": (x, y), "grasper_tip": tuple(gr[0][:2]) if len(gr) > 0 else (np.nan, np.nan),
                   "grasper_tip_2": tuple(gr[1][:2]) if len(gr) > 1 else (np.nan, np.nan)}
            rows[(f.parent.name, img)] = {"src": f.parent / img, "pts": pts}
            n_h += 1
            n_vis += bool(np.isfinite(x))
    cfg = yaml.safe_load((v2 / "config.yaml").read_text(encoding="utf-8"))
    cols = pd.MultiIndex.from_product([[SCORER], BPS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    by_clip, counts = {}, {b: 0 for b in BPS}
    for (clip, img), r in rows.items():
        by_clip.setdefault(clip, []).append((img, r))
    for clip, items in by_clip.items():
        out = dst / "labeled-data" / clip
        out.mkdir(parents=True)
        recs, idx = [], []
        for img, r in items:
            shutil.copy2(r["src"], out / img)
            recs.append([float(v) for b in BPS for v in r["pts"][b]])
            idx.append(("labeled-data", clip, img))
            for b in BPS:
                counts[b] += bool(np.isfinite(r["pts"][b][0]))
        t = pd.DataFrame(recs, index=pd.MultiIndex.from_tuples(idx), columns=cols)
        t.to_hdf(out / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
        t.to_csv(out / f"CollectedData_{SCORER}.csv")
    (dst / "videos").mkdir()
    cfg.update(Task="prism-tips-V3humanlhook", scorer=SCORER, project_path=str(dst),
               video_sets={str(dst / "videos" / f"{c}.mp4"): {"crop": "0, 854, 0, 480"} for c in by_clip})
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    note = {"labels": "L-hook = HUMAN clicks only; graspers = automatic (auto_geo_v1)", "frames": len(rows),
            "v2_grasper_only_frames": n_v2, "human_lhook_frames": n_h, "human_lhook_visible": n_vis,
            "points_per_bodypart": counts, "excluded": "video01 (40 evaluation frames)"}
    (dst / "LABELS_README.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2), "\nproject:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
