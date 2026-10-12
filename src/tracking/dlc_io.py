"""Read DeepLabCut prediction / annotation tables without importing DLC.

DLC writes predictions (``analyze_videos``) and manual labels
(``CollectedData_<scorer>``) as pandas tables with a column MultiIndex:
    single-animal: (scorer, bodyparts, coords)
    multi-animal : (scorer, individuals, bodyparts, coords)
where coords are (x, y, likelihood) for predictions and (x, y) for labels.
Unlabelled points are NaN and are never treated as valid ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_dlc_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DLC file not found: {path}")
    if path.suffix.lower() == ".h5":
        try:
            df = pd.read_hdf(path)
        except ImportError as e:  # pytables missing
            raise ImportError("Reading DLC .h5 needs 'tables' (pip install tables), "
                              "or analyse with save_as_csv=True and pass the .csv") from e
    else:
        # Header: 4 rows (multi-animal) or 3 rows. Row index: 1 column (predictions) or
        # 3 columns ("labeled-data", video, image) as in DLC >= 2.3 CollectedData files.
        # The first header row looks like "scorer,,,name,..." -> the empty cells after the
        # first one are additional index columns.
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].split(",")
        n_index = 1
        while n_index < len(first) and first[n_index].strip() == "":
            n_index += 1
        index_col = list(range(n_index)) if n_index > 1 else 0
        df = None
        for header in ([0, 1, 2, 3], [0, 1, 2]):
            try:
                cand = pd.read_csv(path, header=header, index_col=index_col)
            except (ValueError, IndexError):
                continue
            last = set(cand.columns.get_level_values(-1))
            if last <= {"x", "y", "likelihood"} and {"x", "y"} <= last:
                df = cand
                break
        if df is None:
            raise ValueError(f"{path} does not look like a DeepLabCut table")
    return df


def table_to_keypoints(df: pd.DataFrame, individual: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Return {bodypart: {"x","y","likelihood"}} arrays aligned with df.index.
    For label tables without likelihood, likelihood is 1 where labelled and
    NaN-coordinates rows get likelihood 0."""
    cols = df.columns
    nlev = cols.nlevels
    if nlev == 4:
        inds = list(dict.fromkeys(cols.get_level_values(1)))
        individual = individual or inds[0]
        df = df.xs(individual, axis=1, level=1)
        cols = df.columns
    bodyparts = list(dict.fromkeys(cols.get_level_values(-2)))
    out = {}
    for bp in bodyparts:
        sub = df.xs(bp, axis=1, level=-2)
        sub.columns = sub.columns.get_level_values(-1)
        x = pd.to_numeric(sub["x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)
        if "likelihood" in sub.columns:
            lik = pd.to_numeric(sub["likelihood"], errors="coerce").to_numpy(float)
        else:
            lik = np.where(np.isfinite(x) & np.isfinite(y), 1.0, 0.0)
        out[bp] = {"x": x, "y": y, "likelihood": np.nan_to_num(lik, nan=0.0)}
    return out


def frame_index_from_row_labels(index: pd.Index) -> np.ndarray:
    """DLC prediction tables are indexed 0..N-1; label tables are indexed by
    image path (e.g. labeled-data/video/img00042.png). Extract integers."""
    if np.issubdtype(index.dtype, np.integer):
        return index.to_numpy()
    nums = []
    for v in index:
        v = v[-1] if isinstance(v, tuple) else v
        digits = "".join(ch if ch.isdigit() else " " for ch in Path(str(v)).stem).split()
        nums.append(int(digits[-1]) if digits else -1)
    return np.asarray(nums)


class DLCPredictions:
    """Per-frame lookup into a DLC prediction file for one video."""

    def __init__(self, path: str | Path, individual: str | None = None):
        df = read_dlc_table(path)
        self.path = Path(path)
        self.frames = frame_index_from_row_labels(df.index)
        self.data = table_to_keypoints(df, individual)
        self._row = {int(f): i for i, f in enumerate(self.frames)}
        self.bodyparts = list(self.data)

    def get(self, frame_index: int, bodypart: str) -> tuple[float, float, float] | None:
        if bodypart not in self.data:
            raise KeyError(f"Bodypart '{bodypart}' not in {self.path.name}; available: {self.bodyparts}")
        i = self._row.get(int(frame_index))
        if i is None:
            return None
        d = self.data[bodypart]
        return float(d["x"][i]), float(d["y"][i]), float(d["likelihood"][i])
