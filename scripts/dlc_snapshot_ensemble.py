"""Combine DeepLabCut predictions of several checkpoints of the SAME training run (snapshot ensemble).

Per frame and bodypart: coordinate-wise MEDIAN of x and y and the median likelihood over the checkpoints, so a single
checkpoint that jumps to a wrong spot cannot drag the result (DLC checkpoints swung between epochs here). The last
few checkpoints are used as a fixed choice - no checkpoint is picked by its test score.

  python scripts/dlc_snapshot_ensemble.py --inputs outputs/dlc/v3_probe_ep105 outputs/dlc/v3_probe_ep110 outputs/dlc/v3_model \
      --out outputs/dlc/v3_ensemble
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="folders with <video>DLC*.h5 (one per checkpoint)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    first = Path(a.inputs[0])
    for f in sorted(first.glob("*DLC*.h5")):
        if "_meta" in f.stem:
            continue
        video = f.name.split("DLC")[0]
        tables = [pd.read_hdf(f)]
        for d in a.inputs[1:]:
            hits = [h for h in Path(d).glob(f"{video}DLC*.h5") if "_meta" not in h.stem]
            if not hits:
                raise SystemExit(f"{video}: missing in {d}")
            tables.append(pd.read_hdf(hits[0]))
        base = tables[0].copy()
        # the scorer level differs per checkpoint: match columns by (bodypart, coord) instead
        flat = [t.droplevel(0, axis=1) for t in tables]
        for col in base.columns:
            key = col[1:]
            base[col] = np.nanmedian(np.stack([ft[key].to_numpy(dtype=float) for ft in flat]), axis=0)
        name = f"{video}DLC_ensemble{len(tables)}.h5"
        base.to_hdf(out / name, key="df_with_missing", mode="w")
        base.to_csv(out / name.replace(".h5", ".csv"))
        print("wrote", out / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
