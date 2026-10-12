"""Compare keypoint predictions to human labels.

Ground truth: a DLC CollectedData_<scorer>.csv/.h5 (labelled frames of ONE video).
Predictions : a DLC prediction file for the same video, or a PRISM-Endo
              results.csv (columns <kp>_x, <kp>_y, <kp>_confidence).

  python scripts/evaluate_keypoints.py --gt labeled-data/clip/CollectedData_me.h5 \
      --pred outputs/runs/clip/results.csv --bodyparts lhook_tip --width 854 --height 480
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.keypoints import evaluate_keypoints  # noqa: E402
from src.tracking.dlc_io import DLCPredictions  # noqa: E402


class _PipelineCsvPred:
    def __init__(self, path):
        df = pd.read_csv(path)
        self.df = df.set_index("source_index")

    def get(self, frame, bp):
        if frame not in self.df.index:
            return None
        r = self.df.loc[frame]
        return float(r[f"{bp}_x"]), float(r[f"{bp}_y"]), float(r[f"{bp}_confidence"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--bodyparts", nargs="+", required=True)
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    args = p.parse_args(argv)
    gt = DLCPredictions(args.gt)
    pred = _PipelineCsvPred(args.pred) if Path(args.pred).name == "results.csv" else DLCPredictions(args.pred)
    frames = sorted(int(f) for f in gt.frames if f >= 0)
    size = (args.width, args.height) if args.width and args.height else None
    res = evaluate_keypoints(gt, pred, frames, args.bodyparts, args.min_conf, size)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
