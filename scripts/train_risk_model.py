"""Train the LEARNED risk model (Mode 2). Requires expert risk/event labels.

No such labels exist in this repository; this script refuses to train
without them (see src/risk/learned.py for the label format and checks).

  python scripts/train_risk_model.py \
      --pipeline-csv outputs/runs/caseA/results.csv outputs/runs/caseB/results.csv ... \
      --labels data/risk_labels.csv --out outputs/models/risk_learned [--test-videos caseC caseD]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.learned import attach_labels, features_from_pipeline_csv, train_learned_risk  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline-csv", nargs="+", required=True, help="results.csv files from run_pipeline.py")
    p.add_argument("--labels", required=True, help="expert label CSV (interval or frame form)")
    p.add_argument("--out", default="outputs/models/risk_learned")
    p.add_argument("--test-videos", nargs="*", default=None)
    p.add_argument("--window", type=int, default=25, help="frames in the causal feature window")
    p.add_argument("--min-positive-train", type=int, default=20)
    p.add_argument("--min-positive-test", type=int, default=5)
    args = p.parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print(f"Label file not found: {labels_path}. A learned risk model needs expert-annotated risk "
              f"events; without them use the rule-based research score.")
        return 2
    feats = pd.concat([features_from_pipeline_csv(c, args.window) for c in args.pipeline_csv], ignore_index=True)
    data = attach_labels(feats, pd.read_csv(labels_path))
    print(f"labelled frames: {len(data)}  positives: {int(data['label'].sum())}  "
          f"videos: {sorted(data['video_id'].unique())}")
    try:
        meta = train_learned_risk(data, args.out, args.test_videos, min_positive_train=args.min_positive_train,
                                  min_positive_test=args.min_positive_test, window=args.window)
    except ValueError as e:
        print(f"Cannot train: {e}")
        return 2
    print(json.dumps(meta["metrics_test"], indent=2))
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
