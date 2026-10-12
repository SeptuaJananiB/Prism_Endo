"""Compare the local video01/ with the downloaded CholecSeg8k copy (by SHA-256),
and list which videos are new. Identical videos must not be counted twice or put
in different splits (that would leak test data into training).

  python scripts/compare_datasets.py --old video01 --new data/cholecseg8k
"""

import argparse
import hashlib
import sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="video01")
    ap.add_argument("--new", default="data/cholecseg8k")
    a = ap.parse_args(argv)
    old, new = Path(a.old), Path(a.new)
    vids = sorted({p.parent.parent.name for p in new.rglob("*_endo.png")})
    print(f"videos in {new}: {len(vids)} -> {vids}")
    twin = next((p for p in new.rglob(old.name) if p.is_dir()), None)
    if twin is None:
        print(f"{old.name} not present in {new}: no overlap by name")
        return 0
    old_files = {p.relative_to(old).as_posix(): p for p in old.rglob("*.png")}
    new_files = {p.relative_to(twin).as_posix(): p for p in twin.rglob("*.png")}
    common = sorted(set(old_files) & set(new_files))
    same = sum(sha(old_files[k]) == sha(new_files[k]) for k in common)
    print(f"{old}: {len(old_files)} files | {twin}: {len(new_files)} files | same names: {len(common)} | "
          f"byte-identical: {same} | only local: {len(set(old_files) - set(new_files))} | "
          f"only download: {len(set(new_files) - set(old_files))}")
    print("VERDICT:", "local video01 IS the downloaded video01 (duplicate - use once)" if common and same == len(common)
          == len(old_files) else "local and downloaded video01 differ - inspect before use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
