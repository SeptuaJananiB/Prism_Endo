"""Prepare DeepLabCut projects for manual instrument-tip annotation.

Creates two SEPARATE projects under outputs/dlc_projects/:
  * <name>-train : ~200 diverse frames from TRAIN-split clips (to label and train on)
  * <name>-eval  : ~40 frames from TEST-split clips (to label for held-out evaluation ONLY)
Each gets labeled-data/<clip>/img<NN>.png frames, a selection manifest and
ANNOTATION_GUIDE.md. No labels are created: a human annotates in the DLC GUI.

  python scripts/prepare_dlc_annotation.py --total 200 --eval-total 40
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.segmentation_data import discover_samples, split_samples  # noqa: E402
from src.tracking import dlc_workflow as wf  # noqa: E402
from src.tracking.annotation_prep import (composition, scan_clip, select_frames,  # noqa: E402
                                          write_labeled_data, write_manifest)
from src.utils.config import load_config, resolve_path  # noqa: E402

BODYPARTS = ["lhook_tip", "grasper_tip", "grasper_tip_2"]
GUIDE = resolve_path("docs/DLC_ANNOTATION.md")


def build(name: str, split: str, total: int, cfg: dict, videos_dir: Path, workdir: Path) -> dict:
    samples = discover_samples(resolve_path(cfg["data"]["root"]))
    clips = sorted({s.image.parent for s in split_samples(samples, cfg["data"]["splits"])[split]})
    scanned = {c.name: scan_clip(c) for c in clips}
    frames = select_frames(scanned, total)
    videos = [videos_dir / f"{c}.mp4" for c in sorted({f.clip for f in frames})]
    missing = [v for v in videos if not v.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing videos {missing}; run scripts/make_video_from_frames.py --all video01")
    cfg_path = wf.create_project(name, "annotator", videos, workdir, BODYPARTS, numframes2pick=total)
    project = cfg_path.parent
    images = {}
    for v in videos:
        sel = [f for f in frames if f.clip == v.stem]
        for f, p in zip(sorted(sel, key=lambda f: f.index), write_labeled_data(project, project / "videos" / v.name, sel)):
            images[(f.clip, f.index)] = p
    write_manifest(project / "selection_manifest.csv", frames, images)
    if GUIDE.is_file():
        shutil.copy(GUIDE, project / "ANNOTATION_GUIDE.md")
    comp = composition(frames)
    comp.update(split=split, config=str(cfg_path))
    (project / "selection_summary.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    return comp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="prism-tips")
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--eval-total", type=int, default=40)
    p.add_argument("--config", default="configs/seg_base.yaml", help="provides data root and splits")
    p.add_argument("--videos", default="Videos")
    p.add_argument("--workdir", default="outputs/dlc_projects")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    for suffix, split, n in (("train", "train", args.total), ("eval", "test", args.eval_total)):
        comp = build(f"{args.name}-{suffix}", split, n, cfg, resolve_path(args.videos), resolve_path(args.workdir))
        print(f"\n[{suffix}] {comp['frames']} frames from {split} clips -> {comp['config']}")
        print(f"  with L-hook: {comp['with_l_hook']}  with grasper: {comp['with_grasper']}  "
              f"with two graspers: {comp['with_two_graspers']}")
        print("  per clip:", comp["per_clip"])
    print("\nNo labels were created. Next: label the frames in the DLC GUI (docs/DLC_ANNOTATION.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
