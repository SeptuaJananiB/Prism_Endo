"""Run the complete PRISM-Endo pipeline on a video or an image-sequence folder.

  python run_pipeline.py --input Videos/video01_14859.mp4
  python run_pipeline.py --input video01/video01_28580 --out outputs/runs/clip28580
  python run_pipeline.py --input my.mp4 keypoints.source=dlc keypoints.dlc_predictions=path/myDLC.h5
  python run_pipeline.py --input my.mp4 tissue.checkpoint=null       # instrument-only mode

Outputs (in --out, default outputs/runs/<input name>/):
  annotated.mp4 (+ annotated_h264.mp4 for browsers), results.csv, summary.json
"""

import argparse
import json
import sys

from src.pipeline.pipeline import PrismPipeline
from src.utils.config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="video file or folder of frames")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = p.parse_args(argv)

    overrides = list(args.overrides)
    if args.max_frames:
        overrides.append(f"input.max_frames={args.max_frames}")
    cfg = load_config(args.config, overrides)
    pipe = PrismPipeline(cfg)

    def progress(i, n):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"  frame {i}/{n or '?'}", flush=True)

    summary = pipe.run(args.input, args.out, progress)
    print(json.dumps({k: summary[k] for k in ("frames_processed", "keypoint_source", "tissue_model",
                                              "risk_category_counts", "runtime", "outputs")}, indent=2))
    for w in summary["warnings"]:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
