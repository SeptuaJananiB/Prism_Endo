"""Convert DSAD multi_organ parquet (Hugging Face nct-tso/dsad, CC BY 4.0) to image/label PNGs in PRISM tissue classes.

Mapping onto the 12-class CholecSeg8k head (class indices of outputs/models/tissue_seg_multivideo):
  liver -> liver(2); colon, small_intestine, stomach -> gastrointestinal_tract(3); abdominal_wall -> abdominal_wall(1)
  every other pixel (pancreas, spleen, unannotated fat/instruments/...) -> 255 = ignored in the loss (partial labels).
Splits are DSAD's own (by surgery). Output: data/dsad/png/{split}/{surgery}_{frame}_img.png / _lab.png at 448x256.

  python scripts/prepare_dsad.py
"""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data/dsad", ROOT / "data/dsad/png"
MAP = {"mask_liver": 2, "mask_colon": 3, "mask_small_intestine": 3, "mask_stomach": 3, "mask_abdominal_wall": 1}
W, H = 448, 256


def dec(cell, flag):
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return cv2.imdecode(np.frombuffer(b, np.uint8), flag)


def main():
    stats = {}
    for split in ("train", "validation", "test"):
        f = pq.ParquetFile(SRC / f"multi_organ_{split}.parquet")
        od = OUT / split
        od.mkdir(parents=True, exist_ok=True)
        n, surg, pix = 0, set(), np.zeros(256, np.int64)
        for batch in f.iter_batches(batch_size=16):
            for r in batch.to_pylist():
                img = dec(r["image"], cv2.IMREAD_COLOR)
                lab = np.full(img.shape[:2], 255, np.uint8)
                for k, c in MAP.items():
                    if r.get(k) is not None:
                        m = dec(r[k], cv2.IMREAD_GRAYSCALE)
                        lab[m > 127] = c
                stem = f"{r['surgery_id']}_{r['frame_id']}"
                cv2.imwrite(str(od / f"{stem}_img.png"), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
                lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(od / f"{stem}_lab.png"), lab)
                pix += np.bincount(lab.ravel(), minlength=256)
                n += 1
                surg.add(str(r["surgery_id"]))
        stats[split] = {"frames": n, "surgeries": sorted(surg),
                        "pixel_frac": {str(c): round(float(pix[c] / pix.sum()), 4) for c in (1, 2, 3, 255)}}
        print(split, stats[split], flush=True)
    (OUT / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
