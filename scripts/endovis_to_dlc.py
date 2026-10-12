"""Convert EndoVis'15 joint annotations into a SEPARATE DeepLabCut project (robotic instruments).

Leakage-free split by whole sequence: --train-seqs (default 1 2 3) go into the DLC project;
--heldout-seqs (default 4) are never added to it and are used only by scripts/endovis_evaluate.py.
Frames flagged with duplicate (ambiguous tool identity) entries are excluded, not repaired.
Joint names are source names (tool1_LeftClasperPoint ...), not PRISM lhook/grasper tips.

  python scripts/endovis_to_dlc.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from scripts.endovis_validate import ANN, TRAIN  # noqa: E402
from src.external.endovis import bodyparts, load_labels, to_dlc_table  # noqa: E402
from src.tracking import dlc_workflow as wf  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-seqs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--heldout-seqs", type=int, nargs="+", default=[4])
    ap.add_argument("--workdir", default="outputs/dlc_projects")
    a = ap.parse_args(argv)
    assert not set(a.train_seqs) & set(a.heldout_seqs), "a sequence cannot be in train and held-out"
    frames = []
    for js in sorted((ANN / "train_labels").glob("*.json")):
        frames += load_labels(js, "train")[0]
    excluded = [f for f in frames if f.points and any(i.startswith("duplicate") for i in f.issues)]
    keep = [f for f in frames if f.points and f not in excluded]
    videos = {s: TRAIN / f"Dataset{s}" / "Video.avi" for s in a.train_seqs}
    # DLC copies videos by name; give each a unique stem.
    staged = []
    stage = Path(a.workdir) / "_endovis_video_stage"
    stage.mkdir(parents=True, exist_ok=True)
    for s, v in videos.items():
        dst = stage / f"endovis_seq{s}.avi"
        if not dst.exists():
            dst.write_bytes(v.read_bytes())
        staged.append(dst)
    cfg = wf.create_project("endovis-robotic", "du2018", staged, a.workdir, bodyparts(), numframes2pick=20)
    proj = cfg.parent
    import yaml

    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    scorer = c["scorer"]
    summary = {"project": str(proj), "train_seqs": a.train_seqs, "heldout_seqs": a.heldout_seqs,
               "excluded_ambiguous_frames": [(f.seq, f.frame_number) for f in excluded], "per_seq": {}}
    for s in a.train_seqs:
        stem = f"endovis_seq{s}"
        seq_frames = [f for f in keep if f.seq == s]
        cap = cv2.VideoCapture(str(proj / "videos" / f"{stem}.avi"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        df = to_dlc_table(seq_frames, scorer, stem, n)
        out = proj / "labeled-data" / stem
        out.mkdir(parents=True, exist_ok=True)
        for f, img_name in zip(sorted(seq_frames, key=lambda f: f.frame_number), [i[2] for i in df.index]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f.frame_number - 1)
            ok, img = cap.read()
            if not ok:
                raise IOError(f"cannot decode {stem} frame {f.frame_number}")
            cv2.imwrite(str(out / img_name), img)
        cap.release()
        df.to_hdf(out / f"CollectedData_{scorer}.h5", key="df_with_missing", mode="w")
        df.to_csv(out / f"CollectedData_{scorer}.csv")
        summary["per_seq"][s] = {"frames": len(df), "labelled_points": int(df.notna().sum().sum() // 2)}
        print(f"Dataset{s}: {len(df)} frames -> {out}")
    (proj / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"excluded ambiguous frames: {summary['excluded_ambiguous_frames']}\nproject config: {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
