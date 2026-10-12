"""Count how many train / val / test frames contain each class of a checkpoint
and write class_support.json next to it (read by the pipeline to flag
proximity targets the model never learned or was never tested on).

  python scripts/class_support.py --config configs/tissue_seg.yaml --checkpoint outputs/models/tissue_seg/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.tissue.support import SUPPORT_FILE, compute_class_support, load_support, reliability  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--min-pixels", type=int, default=50, help="pixels for a class to count as present in a frame")
    args = p.parse_args(argv)
    ck_path = resolve_path(args.checkpoint)
    meta = torch.load(str(ck_path), map_location="cpu", weights_only=True)["meta"]
    cfg = load_config(args.config)
    support = compute_class_support(meta, cfg["data"], args.min_pixels)
    (ck_path.parent / SUPPORT_FILE).write_text(json.dumps(support, indent=2), encoding="utf-8")
    _, metrics = load_support(ck_path)
    print(f"{'class':24s} {'train':>6s} {'val':>5s} {'test':>5s}  reliability")
    for name in meta["class_names"][1:]:
        f = {s: support["splits"].get(s, {}).get("frames_with_class", {}).get(name, 0) for s in ("train", "val", "test")}
        r = reliability(support, metrics, name)
        print(f"{name:24s} {f['train']:6d} {f['val']:5d} {f['test']:5d}  {r['reliability']}: {r['note']}")
    print(f"wrote {ck_path.parent / SUPPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
