"""DeepLabCut keypoint workflow for PRISM-Endo (DLC 3.x, PyTorch engine).

Run inside the project venv that has DeepLabCut (.venv). Steps:

  # 0) DLC needs videos; build them from CholecSeg8k clip folders
  python scripts/make_video_from_frames.py --all video01 --out-dir Videos

  # 1) create a project whose bodyparts are the configured tips
  python scripts/dlc_workflow.py create --name prism-tips --experimenter me \
      --videos Videos/video01_00240.mp4 Videos/video01_16345.mp4 Videos/video01_28660.mp4

  # 2) pick diverse frames to label (k-means)
  python scripts/dlc_workflow.py extract --config <project>/config.yaml

  # 3) label them by hand (needs the GUI: pip install "deeplabcut[gui]==3.0.2")
  python scripts/dlc_workflow.py label --config <project>/config.yaml

  # 4) build the training set, 5) train, 6) evaluate
  python scripts/dlc_workflow.py build --config <project>/config.yaml
  python scripts/dlc_workflow.py train --config <project>/config.yaml --epochs 200
  python scripts/dlc_workflow.py evaluate --config <project>/config.yaml

  # 7) predict on a video -> <video>DLC_*.h5/.csv used by run_pipeline.py
  python scripts/dlc_workflow.py analyze --config <project>/config.yaml \
      --video Videos/video01_14859.mp4 --out outputs/dlc/video01_14859
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking import dlc_workflow as wf  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", default="prism-tips")
    c.add_argument("--experimenter", default="annotator")
    c.add_argument("--videos", nargs="+", required=True)
    c.add_argument("--workdir", default="outputs/dlc_projects")
    c.add_argument("--pipeline-config", default="configs/pipeline.yaml",
                   help="bodyparts are taken from keypoints.points[].dlc_bodypart")
    c.add_argument("--numframes", type=int, default=20, help="frames to extract per video")
    for name in ("extract", "label", "build", "evaluate", "status"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        if name == "build":
            s.add_argument("--net", default="resnet_50", help="DLC PyTorch net_type, e.g. resnet_50, hrnet_w32")
        if name in ("build", "status"):
            s.add_argument("--min-frames", type=int, default=100)
            s.add_argument("--min-per-bodypart", type=int, default=30)
    t = sub.add_parser("train")
    t.add_argument("--min-frames", type=int, default=100)
    t.add_argument("--min-per-bodypart", type=int, default=30)
    t.add_argument("--config", required=True)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--device", default="auto")
    a = sub.add_parser("analyze")
    a.add_argument("--config", required=True)
    a.add_argument("--video", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="auto")
    a.add_argument("--no-labeled-video", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "create":
        pts = load_config(args.pipeline_config)["keypoints"]["points"]
        bodyparts = [d.get("dlc_bodypart") or d["name"] for d in pts]
        cfg = wf.create_project(args.name, args.experimenter, args.videos, args.workdir, bodyparts, args.numframes)
        print(f"Created DLC project: {cfg}\nBodyparts: {bodyparts}")
    elif args.cmd == "extract":
        wf.extract_frames(args.config)
    elif args.cmd == "label":
        wf.label_frames(args.config)
    elif args.cmd == "status":
        st = wf.label_status(args.config)
        print(json.dumps(st, indent=2))
        try:
            wf.require_labels(args.config, args.min_frames, args.min_per_bodypart)
            print("Enough human labels to train.")
        except RuntimeError as e:
            print(e)
            return 2
    elif args.cmd in ("build", "train"):
        try:  # never build/train on missing or too few human labels
            wf.require_labels(args.config, args.min_frames, args.min_per_bodypart)
        except RuntimeError as e:
            print(e)
            return 2
        if args.cmd == "build":
            wf.build_dataset(args.config, args.net)
        else:
            wf.train(args.config, args.epochs, args.batch_size, args.device)
    elif args.cmd == "evaluate":
        wf.evaluate(args.config)
    elif args.cmd == "analyze":
        out = wf.analyze(args.config, args.video, args.out, args.device, labeled_video=not args.no_labeled_video)
        print(f"DLC predictions: {out}\nUse with: python run_pipeline.py --input {args.video} "
              f"keypoints.source=dlc keypoints.dlc_predictions={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
