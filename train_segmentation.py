"""Train an instrument or tissue segmentation model.

Examples
--------
  # quick smoke test (2 batches, 1 epoch, small subsets)
  python train_segmentation.py --config configs/instrument_seg.yaml --smoke

  # full run (CPU: roughly minutes per epoch; GPU: much faster)
  python train_segmentation.py --config configs/instrument_seg.yaml
  python train_segmentation.py --config configs/tissue_seg.yaml train.epochs=12

  # resume
  python train_segmentation.py --config configs/instrument_seg.yaml \
      --resume outputs/models/instrument_seg/last.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.segmentation.train import train
from src.utils.config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="path to last.pt to continue training")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to check the code path (not a real training run)")
    p.add_argument("--max-restarts", type=int, default=0,
                   help="run training in a child process and, if it crashes (e.g. out of memory), "
                        "resume from last.pt up to N times")
    p.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = p.parse_args(argv)
    if args.max_restarts > 0:
        return _run_with_restarts(args)

    overrides = list(args.overrides)
    if args.smoke:
        overrides = ["train.epochs=1", "train.max_train_batches=2", "train.limit_train_samples=16",
                     "train.limit_val_samples=8", "output_dir=outputs/smoke/seg"] + overrides
    cfg = load_config(args.config, overrides)
    result = train(cfg, resume=args.resume)
    print(f"Done: {result['epochs_completed']} epochs, best val score={result['best_score']:.4f}, "
          f"checkpoints in {result['output_dir']}")
    return 0


def _run_with_restarts(args) -> int:
    from src.utils.config import resolve_path

    cfg = load_config(args.config, args.overrides)
    last = resolve_path(cfg["output_dir"]) / "last.pt"
    resume = args.resume
    for attempt in range(args.max_restarts + 1):
        cmd = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), "--config", args.config]
        if resume:
            cmd += ["--resume", str(resume)]
        if args.smoke:
            cmd.append("--smoke")
        code = subprocess.call(cmd + list(args.overrides))
        if code == 0:
            return 0
        print(f"[train_segmentation] attempt {attempt + 1} exited with code {code}", flush=True)
        # Resume from the newest checkpoint if one exists, else retry from scratch.
        resume = last if last.is_file() else args.resume
    return code


if __name__ == "__main__":
    sys.exit(main())
