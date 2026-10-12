"""Risk / near-miss EVENT labels: schema, validation, inter-rater agreement, training gate.

No risk labels exist in this repository. This module checks label files that a
surgeon or trained annotator creates (with src/ui/annotate_events.py or by hand)
BEFORE they are used to train or evaluate the learned risk model
(src/risk/learned.py). It never creates labels.

Label file = the interval CSV accepted by learned.attach_labels, with optional
extra columns (backward compatible):

    video_id,start_s,end_s,label[,event_type,severity,annotator,confidence,notes,blind,created_at]

Schema and PROPOSED event definitions: configs/risk_events.yaml; guide:
docs/RISK_ANNOTATION.md. The legacy frame form (video_id,source_index,label)
gets basic checks only.

Semantics: a video that appears in a label file is annotated COMPLETELY by each
annotator who has rows for it; every frame not inside a label-1 interval is
negative. A `review_complete` row (label 0, [0, duration]) records that an
annotator finished a video - the only way to record a video with no events.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "configs" / "risk_events.yaml"
INTERVAL_REQUIRED = ["video_id", "start_s", "end_s", "label"]
FRAME_REQUIRED = ["video_id", "source_index", "label"]
ANNOTATION_COLUMNS = ["video_id", "start_s", "end_s", "label", "event_type", "severity", "annotator",
                      "confidence", "notes", "blind", "created_at"]
EPS = 1e-6
FALLBACK_NOTE = ("PRISM keeps using the rule-based research score (Mode 1) - an uncalibrated heuristic that has "
                 "not been validated against any risk or near-miss labels. To create labels see "
                 "docs/RISK_ANNOTATION.md.")


# --------------------------------------------------------------------------- schema

def load_schema(path: str | Path | None = None) -> dict:
    with open(path or SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def positive_event_types(schema: dict) -> list[str]:
    return [k for k, v in schema["event_types"].items() if int(v.get("label", 1)) == 1]


def marker_type(schema: dict) -> str:
    return schema.get("coverage_marker", "review_complete")


# --------------------------------------------------------------------------- reading / writing

def expand_label_paths(items) -> list[Path]:
    """Files, directories (-> their *.csv) and glob patterns (PowerShell does not expand them)."""
    out = []
    for it in items or []:
        p = Path(it)
        if p.is_dir():
            out += sorted(p.glob("*.csv"))
        elif any(ch in str(it) for ch in "*?["):
            out += [Path(g) for g in sorted(glob.glob(str(it)))]
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def read_label_files(paths) -> pd.DataFrame:
    """Read label CSVs as text (no type guessing) and tag each row with its file/line.
    A missing `annotator` column is filled with the file stem (flagged in `_annotator_inferred`)."""
    frames = []
    for p in paths:
        p = Path(p)
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        df.columns = [str(c).strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        inferred = "annotator" not in df.columns
        if inferred:
            df["annotator"] = p.stem
        df["_annotator_inferred"] = inferred
        df["_file"] = p.name
        df["_row"] = np.arange(len(df)) + 2  # CSV line number (header is line 1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _parse_bool(v):
    s = str(v).strip().lower()
    return True if s in ("true", "1", "yes") else False if s in ("false", "0", "no") else None


def normalize_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed copy: times/label/severity/confidence numeric (NaN if invalid), text stripped."""
    df = raw.copy()
    for c in ("start_s", "end_s", "label", "severity", "confidence", "source_index"):
        if c in df.columns:
            df[c] = _num(df[c].astype(str))
    for c in ("video_id", "event_type", "annotator", "notes"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "event_type" not in df.columns:
        df["event_type"] = ""
    if "blind" in df.columns:
        df["blind"] = df["blind"].map(_parse_bool)
    df.loc[df["annotator"] == "", "annotator"] = "unknown"
    return df


def safe_annotator_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip()).strip("_")


def annotation_path(directory: str | Path, annotator: str) -> Path:
    return Path(directory) / f"risk_events_{safe_annotator_id(annotator)}.csv"


def read_annotation_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ANNOTATION_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_annotation(path: str | Path, row: dict) -> pd.DataFrame:
    """Append one row (keys from ANNOTATION_COLUMNS) to an annotator's CSV; returns the new table."""
    path = Path(path)
    df = read_annotation_file(path)
    row = {c: row.get(c, "") for c in ANNOTATION_COLUMNS}
    row.setdefault("created_at", "")
    if not row["created_at"]:
        row["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = pd.concat([df, pd.DataFrame([{k: ("" if v is None else str(v)) for k, v in row.items()}])],
                    ignore_index=True)
    _atomic_write(new, path)
    return new


def delete_annotation_rows(path: str | Path, indices) -> pd.DataFrame:
    path = Path(path)
    df = read_annotation_file(path).drop(index=list(indices), errors="ignore").reset_index(drop=True)
    _atomic_write(df, path)
    return df


# --------------------------------------------------------------------------- video durations

def durations_from_results_csv(path: str | Path) -> dict:
    """Processed time range per video_id from a pipeline results.csv."""
    df = pd.read_csv(path, usecols=["video_id", "timestamp_s"], dtype={"video_id": str})
    out = {}
    for vid, g in df.groupby("video_id"):
        t = np.sort(g["timestamp_s"].to_numpy(float))
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
        dt = dt if dt > 0 else 0.04
        out[str(vid)] = {"duration_s": float(t[-1] + dt), "last_frame_s": float(t[-1]), "fps": 1.0 / dt,
                         "source": f"pipeline results ({Path(path).parent.name}/{Path(path).name})"}
    return out


def duration_from_video(path: str | Path) -> dict | None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if n <= 0:
        return None
    fps = fps if fps > 1 else 25.0
    return {"duration_s": n / fps, "last_frame_s": (n - 1) / fps, "fps": fps, "source": f"video file ({Path(path).name})"}


def duration_from_frame_folder(folder: str | Path, fps: float = 25.0) -> dict | None:
    """Same timestamps as the pipeline: (frame number - first frame number) / fps."""
    from src.utils.video_io import _frame_number, list_image_frames

    files = list_image_frames(Path(folder))
    nums = [_frame_number(p) for p in files]
    if not files:
        return None
    if all(n is not None for n in nums):
        last = (max(nums) - min(nums)) / fps
    else:
        last = (len(files) - 1) / fps
    return {"duration_s": last + 1.0 / fps, "last_frame_s": last, "fps": fps,
            "source": f"frame folder ({Path(folder).name})"}


def collect_durations(results_csvs=(), runs_dir=None, videos_dirs=(), fps: float = 25.0) -> dict:
    """video_id -> {duration_s, last_frame_s, fps, source}. When several sources know a
    video (e.g. a run limited by --max-frames and the full video file), the longest wins."""
    found: list[tuple[str, dict]] = []

    def add(d: dict):
        found.extend(d.items())

    csvs = [Path(c) for c in results_csvs or []]
    if runs_dir and Path(runs_dir).is_dir():
        csvs += sorted(Path(runs_dir).glob("*/results.csv"))
    for c in csvs:
        try:
            d = durations_from_results_csv(c)
        except (OSError, ValueError, KeyError):
            continue
        add(d)
        summ = c.parent / "summary.json"
        if summ.is_file() and len(d) == 1:
            try:
                inp = json.loads(summ.read_text(encoding="utf-8")).get("input")
            except (OSError, ValueError):
                inp = None
            if inp:
                ip = Path(inp) if Path(inp).is_absolute() else ROOT / inp
                vid = next(iter(d))
                full = duration_from_video(ip) if ip.is_file() else \
                    duration_from_frame_folder(ip, fps) if ip.is_dir() else None
                if full:
                    found.append((vid, full))
    from src.utils.video_io import VIDEO_EXTS

    for vd in videos_dirs or []:
        vd = Path(vd)
        if not vd.is_dir():
            continue
        for p in sorted(vd.iterdir()):
            info = duration_from_video(p) if p.is_file() and p.suffix.lower() in VIDEO_EXTS else \
                duration_from_frame_folder(p, fps) if p.is_dir() else None
            if info:
                found.append((p.name if p.is_dir() else p.stem, info))
    out: dict = {}
    for vid, info in found:
        if vid not in out or info["duration_s"] > out[vid]["duration_s"] + EPS:
            out[vid] = info
    return out


# --------------------------------------------------------------------------- interval helpers

def merge_intervals(intervals) -> list[tuple[float, float]]:
    iv = sorted((float(s), float(e)) for s, e in intervals)
    out: list[list[float]] = []
    for s, e in iv:
        if out and s <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def union_length(intervals, lo: float | None = None, hi: float | None = None) -> float:
    total = 0.0
    for s, e in merge_intervals(intervals):
        s = s if lo is None else max(s, lo)
        e = e if hi is None else min(e, hi)
        total += max(0.0, e - s)
    return total


def interval_mask(t, intervals) -> np.ndarray:
    """Frames whose time lies inside any [start, end] (inclusive, like attach_labels)."""
    t = np.asarray(t, float)
    m = np.zeros(len(t), bool)
    for s, e in intervals:
        m |= (t >= s - EPS) & (t <= e + EPS)
    return m


def events_by_video(df: pd.DataFrame, label: int = 1) -> dict[str, list[tuple[float, float]]]:
    sel = df[df["label"].astype(float) == label]
    return {str(v): list(zip(g["start_s"].astype(float), g["end_s"].astype(float)))
            for v, g in sel.groupby(sel["video_id"].astype(str))}


def merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and merge overlapping/touching intervals of the SAME
    (video, annotator, event type, label). severity -> max, confidence -> min, notes joined."""
    df = normalize_labels(df) if df["start_s"].dtype == object else df.copy()
    keys = [k for k in ("video_id", "annotator", "event_type", "label") if k in df.columns]
    rows = []
    for _, g in df.sort_values("start_s").groupby(keys, dropna=False, sort=True):
        cur = None
        for _, r in g.iterrows():
            r = r.to_dict()
            if cur is not None and r["start_s"] <= cur["end_s"] + EPS:
                cur["end_s"] = max(cur["end_s"], r["end_s"])
                if "severity" in cur:
                    cur["severity"] = _nan_agg(max, cur["severity"], r["severity"])
                if "confidence" in cur:
                    cur["confidence"] = _nan_agg(min, cur["confidence"], r["confidence"])
                if "notes" in cur and r.get("notes") and r["notes"] not in cur["notes"]:
                    cur["notes"] = " | ".join(x for x in (cur["notes"], r["notes"]) if x)
                if "blind" in cur:
                    cur["blind"] = bool(cur["blind"]) and bool(r["blind"]) \
                        if cur["blind"] is not None and r["blind"] is not None else None
            else:
                if cur is not None:
                    rows.append(cur)
                cur = r
        if cur is not None:
            rows.append(cur)
    out = pd.DataFrame(rows)
    cols = [c for c in ANNOTATION_COLUMNS if c in out.columns]
    return to_label_table(out[cols].sort_values(["video_id", "annotator", "start_s"]).reset_index(drop=True))


def _nan_agg(fn, a, b):
    vals = [v for v in (a, b) if v is not None and np.isfinite(v)]
    return fn(vals) if vals else np.nan


def to_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format a normalized label table for writing: integer label/severity/confidence,
    true/false blind, no helper (_*) columns."""
    out = df[[c for c in df.columns if not c.startswith("_")]].copy()
    for c in ("label", "severity", "confidence"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    if "blind" in out.columns:
        out["blind"] = out["blind"].map(lambda v: "" if v is None or v != v else str(bool(v)).lower())
    return out


# --------------------------------------------------------------------------- agreement

def frame_kappa(a: np.ndarray, b: np.ndarray) -> dict:
    """Cohen's kappa for two binary frame series (None if undefined, e.g. both all-negative)."""
    from sklearn.metrics import cohen_kappa_score

    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return {"kappa": None, "percent_agreement": None, "frames": 0, "positives_a": 0, "positives_b": 0}
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if pe >= 1 - EPS else float(cohen_kappa_score(a, b, labels=[0, 1]))
    return {"kappa": kappa, "percent_agreement": po, "frames": n,
            "positives_a": int(a.sum()), "positives_b": int(b.sum())}


def match_events(a: list[dict], b: list[dict]) -> dict:
    """One-to-one event matching: two events match if their intervals overlap (inclusive);
    greedy by IoU. Events are dicts with start_s, end_s and optionally event_type."""
    pairs = []
    for i, ea in enumerate(a):
        for j, eb in enumerate(b):
            inter = min(ea["end_s"], eb["end_s"]) - max(ea["start_s"], eb["start_s"])
            if inter >= -EPS:
                union = max(ea["end_s"], eb["end_s"]) - min(ea["start_s"], eb["start_s"])
                pairs.append((max(inter, 0.0) / union if union > 0 else 1.0, i, j))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_a, used_b, matches = set(), set(), []
    for iou, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((i, j, iou))
    m, na, nb = len(matches), len(a), len(b)
    same_type = [a[i].get("event_type", "") == b[j].get("event_type", "") for i, j, _ in matches]
    return {
        "events_a": na, "events_b": nb, "matched": m,
        "event_f1": 2 * m / (na + nb) if na + nb else None,
        "type_agreement": float(np.mean(same_type)) if matches else None,
        "mean_iou": float(np.mean([x[2] for x in matches])) if matches else None,
        "mean_abs_start_diff_s": float(np.mean([abs(a[i]["start_s"] - b[j]["start_s"]) for i, j, _ in matches]))
        if matches else None,
        "mean_abs_end_diff_s": float(np.mean([abs(a[i]["end_s"] - b[j]["end_s"]) for i, j, _ in matches]))
        if matches else None,
        "matches": [(i, j, round(iou, 4)) for i, j, iou in matches],
        "unmatched_a": [i for i in range(na) if i not in used_a],
        "unmatched_b": [j for j in range(nb) if j not in used_b],
    }


def _video_grid(vid: str, rows: pd.DataFrame, durations: dict, fps: float) -> np.ndarray:
    info = durations.get(vid)
    if info:
        step = 1.0 / float(info.get("fps") or fps)
        last = info["last_frame_s"]
    else:
        step = 1.0 / fps
        last = float(rows["end_s"].max())
    return np.arange(int(np.floor(last / step + EPS)) + 1) * step


def pairwise_agreement(df: pd.DataFrame, durations: dict | None = None, fps: float = 25.0,
                       marker: str = "review_complete") -> list[dict]:
    """Agreement for every pair of annotators that labelled the same video(s):
    frame-level Cohen's kappa (any event vs none, pooled over shared videos, time grid at
    the video's frame rate) and event-level matching (overlap = match)."""
    durations = durations or {}
    out = []
    annotators = sorted(df["annotator"].unique())
    for a, b in itertools.combinations(annotators, 2):
        shared = sorted(set(df.loc[df.annotator == a, "video_id"]) & set(df.loc[df.annotator == b, "video_id"]))
        if not shared:
            continue
        ya, yb, per_video, disagreements = [], [], [], []
        tot = {"events_a": 0, "events_b": 0, "matched": 0, "same_type": 0}
        for vid in shared:
            rows = df[df.video_id == vid]
            t = _video_grid(vid, rows, durations, fps)
            ev = {}
            for who in (a, b):
                r = rows[(rows.annotator == who) & (rows.label == 1) & (rows.event_type != marker)]
                ev[who] = r.sort_values("start_s")[["start_s", "end_s", "event_type"]].to_dict("records")
            ma, mb = interval_mask(t, [(e["start_s"], e["end_s"]) for e in ev[a]]), \
                interval_mask(t, [(e["start_s"], e["end_s"]) for e in ev[b]])
            ya.append(ma)
            yb.append(mb)
            k = frame_kappa(ma, mb)
            em = match_events(ev[a], ev[b])
            tot["events_a"] += em["events_a"]
            tot["events_b"] += em["events_b"]
            tot["matched"] += em["matched"]
            tot["same_type"] += sum(ev[a][i]["event_type"] == ev[b][j]["event_type"] for i, j, _ in em["matches"])
            per_video.append({"video_id": vid, "kappa": k["kappa"], "percent_agreement": k["percent_agreement"],
                              "frames": k["frames"], **{x: em[x] for x in ("events_a", "events_b", "matched",
                                                                           "event_f1", "mean_iou")}})
            for who, idx in ((a, em["unmatched_a"]), (b, em["unmatched_b"])):
                for i in idx:
                    disagreements.append({"video_id": vid, "only_annotator": who, **ev[who][i]})
        k = frame_kappa(np.concatenate(ya), np.concatenate(yb))
        n_ev = tot["events_a"] + tot["events_b"]
        out.append({
            "annotators": [a, b], "videos": shared, "frame_kappa": k["kappa"],
            "percent_agreement": k["percent_agreement"], "frames": k["frames"],
            "events_a": tot["events_a"], "events_b": tot["events_b"], "events_matched": tot["matched"],
            "event_f1": 2 * tot["matched"] / n_ev if n_ev else None,
            "type_agreement": tot["same_type"] / tot["matched"] if tot["matched"] else None,
            "per_video": per_video, "disagreements": disagreements,
        })
    return out


# --------------------------------------------------------------------------- validation

@dataclass
class ValidationReport:
    form: str = "interval"
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    coverage: list = field(default_factory=list)
    agreement: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"ok": self.ok, "form": self.form, "errors": self.errors, "warnings": self.warnings,
                "stats": self.stats, "coverage": self.coverage, "agreement": self.agreement}

    def format(self, max_items: int = 40) -> str:
        lines = [f"Label validation: {'VALID' if self.ok else 'INVALID'} ({self.form} form) - "
                 f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for title, items in (("ERRORS", self.errors), ("WARNINGS", self.warnings)):
            if items:
                lines.append(f"{title}:")
                lines += [f"  - {m}" for m in items[:max_items]]
                if len(items) > max_items:
                    lines.append(f"  ... and {len(items) - max_items} more")
        s = self.stats
        if s:
            lines.append("SUMMARY:")
            keys = ("rows_total", "rows_invalid", "videos", "annotators", "events", "events_per_video_mean", "total_positive_time_s",
                    "prevalence", "by_event_type", "by_annotator", "non_blind_rows")
            lines += [f"  {k}: {s[k]}" for k in keys if k in s]
        if self.coverage:
            done = sum(1 for c in self.coverage if c["complete"])
            lines.append(f"COVERAGE: {done}/{len(self.coverage)} (video, annotator) pairs marked completely reviewed")
        for ag in self.agreement:
            def f(v):
                return "n/a" if v is None else f"{v:.3f}"
            lines.append(f"AGREEMENT {ag['annotators'][0]} vs {ag['annotators'][1]} on {len(ag['videos'])} video(s): "
                         f"frame kappa={f(ag['frame_kappa'])} (target {f(ag.get('kappa_target'))}), "
                         f"event F1={f(ag['event_f1'])} (target {f(ag.get('event_f1_target'))}), "
                         f"matched {ag['events_matched']} of {ag['events_a']}/{ag['events_b']} events, "
                         f"{len(ag['disagreements'])} unmatched event(s) to adjudicate")
        return "\n".join(lines)


def _where(r) -> str:
    return f"{r['_file']}:{r['_row']}" if "_file" in r else f"row {r.name}"


def validate_labels(raw: pd.DataFrame, schema: dict | None = None, durations: dict | None = None) -> ValidationReport:
    """Validate label rows (as returned by read_label_files). Never modifies the data."""
    schema = schema or load_schema()
    durations = durations or {}
    rep = ValidationReport()
    if raw is None or raw.empty or not [c for c in raw.columns if not c.startswith("_")]:
        rep.errors.append("No label rows found.")
        return rep
    cols = set(raw.columns)
    if {"start_s", "end_s"} <= cols:
        rep.form = "interval"
        missing = [c for c in INTERVAL_REQUIRED if c not in cols]
    elif "source_index" in cols:
        rep.form = "frame"
        missing = [c for c in FRAME_REQUIRED if c not in cols]
    else:
        rep.form = "unknown"
        missing = [c for c in INTERVAL_REQUIRED if c not in cols]
    if missing:
        rep.errors.append(f"Missing required column(s) {missing}. Interval form needs "
                          f"{','.join(INTERVAL_REQUIRED)} (frame form: {','.join(FRAME_REQUIRED)}).")
        return rep
    known = set(INTERVAL_REQUIRED) | set(FRAME_REQUIRED) | set(schema["columns"].get("optional", []))
    extra = sorted(c for c in cols if c not in known and not c.startswith("_"))
    if extra:
        rep.warnings.append(f"Unknown column(s) ignored: {extra}")
    if raw["_annotator_inferred"].any():
        files = sorted(raw.loc[raw["_annotator_inferred"], "_file"].unique())
        rep.warnings.append(f"No 'annotator' column in {files}: annotator set to the file name.")
    df = normalize_labels(raw)
    allowed_labels = set(schema.get("label_values", [0, 1]))
    bad = pd.Series(False, index=df.index)

    for i, r in df.iterrows():
        w = _where(r)
        if not r["video_id"]:
            rep.errors.append(f"{w}: empty video_id")
            bad[i] = True
        if not np.isfinite(r["label"]) or r["label"] not in allowed_labels:
            rep.errors.append(f"{w}: label {raw.at[i, 'label']!r} not in {sorted(allowed_labels)}")
            bad[i] = True
    if rep.form == "frame":
        return _validate_frame_form(df, bad, rep)

    pos_types, mtype = positive_event_types(schema), marker_type(schema)
    types = schema["event_types"]
    sev_keys = {int(k) for k in schema.get("severity", {})}
    conf_keys = {int(k) for k in schema.get("confidence", {})}
    for i, r in df.iterrows():
        w = _where(r)
        s, e = r["start_s"], r["end_s"]
        if not (np.isfinite(s) and np.isfinite(e)):
            rep.errors.append(f"{w}: start_s/end_s not numeric ({raw.at[i, 'start_s']!r}, {raw.at[i, 'end_s']!r})")
            bad[i] = True
            continue
        if s < 0 or e < 0:
            rep.errors.append(f"{w}: negative time ({s}, {e})")
            bad[i] = True
        if s >= e:
            rep.errors.append(f"{w}: start_s ({s}) must be < end_s ({e})")
            bad[i] = True
        et = r["event_type"]
        if et:
            if et not in types:
                rep.errors.append(f"{w}: unknown event_type '{et}' (known: {sorted(types)})")
                bad[i] = True
            elif np.isfinite(r["label"]) and int(r["label"]) != int(types[et].get("label", 1)):
                rep.errors.append(f"{w}: event_type '{et}' requires label {types[et].get('label', 1)}, "
                                  f"got {r['label']:g}")
                bad[i] = True
            elif types[et].get("requires_notes") and not r.get("notes", ""):
                rep.warnings.append(f"{w}: event_type '{et}' should be described in notes")
        elif r["label"] == 1 and "event_type" in raw.columns:
            rep.warnings.append(f"{w}: event without event_type (counted as a generic risk event)")
        for col, keys in (("severity", sev_keys), ("confidence", conf_keys)):
            if col in raw.columns and str(raw.at[i, col]).strip() != "":
                v = r[col]
                if not np.isfinite(v) or int(v) != v or int(v) not in keys:
                    rep.errors.append(f"{w}: {col} {raw.at[i, col]!r} not in {sorted(keys)}")
                    bad[i] = True
        info = durations.get(r["video_id"])
        if info and np.isfinite(e):
            tol = float(schema.get("validation", {}).get("duration_tolerance_s", 0.1))
            if e > info["duration_s"] + tol or s > info["duration_s"]:
                rep.errors.append(f"{w}: interval [{s}, {e}] s outside video '{r['video_id']}' "
                                  f"(duration {info['duration_s']:.2f} s from {info['source']})")
                bad[i] = True

    unknown = sorted(v for v in df["video_id"].unique() if v and v not in durations)
    if unknown:
        rep.warnings.append(f"No duration source (pipeline results.csv / video file) for video_id(s) {unknown}: "
                            f"time range not checked - check the IDs match the pipeline's video_id.")

    key = ["video_id", "start_s", "end_s", "label", "event_type", "annotator"]
    dup = df.duplicated(subset=key, keep="first")
    for i in df.index[dup]:
        rep.errors.append(f"{_where(df.loc[i])}: duplicate row ({df.at[i, 'video_id']}, {df.at[i, 'start_s']}-"
                          f"{df.at[i, 'end_s']}, {df.at[i, 'event_type'] or 'no type'}, {df.at[i, 'annotator']})")
        bad[i] = True

    good = df[~bad]
    ev = good[(good["label"] == 1) & (good["event_type"] != mtype)]
    for (vid, ann), g in ev.groupby(["video_id", "annotator"]):
        g = g.sort_values("start_s")
        recs = g.to_dict("records")
        for x, y in itertools.combinations(recs, 2):
            if min(x["end_s"], y["end_s"]) - max(x["start_s"], y["start_s"]) > EPS:
                msg = (f"{x['_file']}:{x['_row']} and {y['_file']}:{y['_row']}: overlapping intervals for "
                       f"video '{vid}', annotator '{ann}' ([{x['start_s']}, {x['end_s']}] / [{y['start_s']}, {y['end_s']}]")
                if x["event_type"] == y["event_type"]:
                    rep.errors.append(msg + f", same type '{x['event_type'] or 'none'}') - merge them "
                                            f"(validate_risk_labels.py --write-merged)")
                else:
                    rep.warnings.append(msg + f", types {x['event_type']} / {y['event_type']}) - concurrent events")

    rep.coverage = _coverage(good, durations, schema)
    for c in rep.coverage:
        if not c["has_marker"]:
            rep.warnings.append(f"video '{c['video_id']}', annotator '{c['annotator']}': no '{mtype}' row - the "
                                f"video is ASSUMED completely annotated (everything outside events = negative).")
        elif not c["complete"] and c["coverage_fraction"] is not None:
            rep.warnings.append(f"video '{c['video_id']}', annotator '{c['annotator']}': '{mtype}' covers only "
                                f"{100 * c['coverage_fraction']:.0f}% of the video ({c['duration_s']:.2f} s, "
                                f"{c['duration_source']}); the rest would still be treated as negative.")
    if "blind" in good.columns:
        nb = int((good["blind"] == False).sum())  # noqa: E712 - None means unknown
        if nb:
            rep.warnings.append(f"{nb} row(s) were annotated with the PRISM overlay/score visible (blind=false).")

    rep.stats = {"rows_total": len(df), "rows_invalid": int(bad.sum()), **summary_stats(good, durations, schema)}
    ag = schema.get("agreement", {})
    rep.agreement = pairwise_agreement(good, durations, float(ag.get("grid_fps", 25.0)), mtype)
    for a in rep.agreement:
        a["kappa_target"], a["event_f1_target"] = ag.get("frame_kappa_target"), ag.get("event_f1_target")
        names = " vs ".join(a["annotators"])
        if a["frame_kappa"] is not None and a.get("kappa_target") is not None and a["frame_kappa"] < a["kappa_target"]:
            rep.warnings.append(f"Agreement {names}: frame kappa {a['frame_kappa']:.3f} below target "
                                f"{a['kappa_target']} - revise definitions / retrain annotators before scaling up.")
        if a["event_f1"] is not None and a.get("event_f1_target") is not None and a["event_f1"] < a["event_f1_target"]:
            rep.warnings.append(f"Agreement {names}: event F1 {a['event_f1']:.3f} below target {a['event_f1_target']}.")
    return rep


def _validate_frame_form(df: pd.DataFrame, bad: pd.Series, rep: ValidationReport) -> ValidationReport:
    for i, r in df.iterrows():
        v = r["source_index"]
        if not np.isfinite(v) or v < 0 or int(v) != v:
            rep.errors.append(f"{_where(r)}: source_index {v!r} must be a non-negative integer")
            bad[i] = True
    dup = df.duplicated(subset=["video_id", "source_index", "annotator"], keep="first")
    for i in df.index[dup]:
        rep.errors.append(f"{_where(df.loc[i])}: duplicate frame ({df.at[i, 'video_id']}, {df.at[i, 'source_index']})")
    good = df[~bad & ~dup]
    rep.warnings.append("Frame-form labels: event types, overlaps, coverage, agreement and event-level warning "
                        "evaluation need the interval form.")
    rep.stats = {"rows": len(df), "videos": int(good["video_id"].nunique()),
                 "annotators": sorted(good["annotator"].unique()),
                 "positive_frames": int((good["label"] == 1).sum()),
                 "prevalence": float((good["label"] == 1).mean()) if len(good) else None}
    return rep


def _coverage(df: pd.DataFrame, durations: dict, schema: dict) -> list[dict]:
    mtype = marker_type(schema)
    tol = float(schema.get("validation", {}).get("coverage_tolerance_s", 0.1))
    out = []
    for (vid, ann), g in df.groupby(["video_id", "annotator"]):
        marks = list(zip(g.loc[g["event_type"] == mtype, "start_s"], g.loc[g["event_type"] == mtype, "end_s"]))
        info = durations.get(vid)
        frac, complete = None, bool(marks)
        if info and marks:
            last = info["last_frame_s"]
            merged = merge_intervals(marks)
            frac = union_length(marks, 0.0, last) / last if last > 0 else 1.0
            complete = any(s <= tol and e >= last - tol for s, e in merged)
        out.append({"video_id": vid, "annotator": ann, "has_marker": bool(marks), "complete": complete,
                    "coverage_fraction": None if frac is None else round(min(frac, 1.0), 4),
                    "duration_s": info["duration_s"] if info else None,
                    "duration_source": info["source"] if info else None})
    return out


def summary_stats(df: pd.DataFrame, durations: dict | None = None, schema: dict | None = None) -> dict:
    durations = durations or {}
    mtype = marker_type(schema) if schema else "review_complete"
    ev = df[(df["label"] == 1) & (df["event_type"] != mtype)]
    per = []
    for (vid, ann), g in df.groupby(["video_id", "annotator"]):
        e = ev[(ev.video_id == vid) & (ev.annotator == ann)]
        info = durations.get(vid)
        dur = info["duration_s"] if info else None
        pos = union_length(zip(e["start_s"], e["end_s"]), 0.0, dur)
        per.append({"video_id": vid, "annotator": ann, "events": len(e), "positive_time_s": round(pos, 3),
                    "duration_s": dur, "prevalence": round(pos / dur, 4) if dur else None})
    known = [p for p in per if p["duration_s"]]
    counts = [p["events"] for p in per]
    out = {
        "rows": len(df), "videos": int(df["video_id"].nunique()), "annotators": sorted(df["annotator"].unique()),
        "events": len(ev),
        "events_per_video_mean": round(float(np.mean(counts)), 3) if counts else None,
        "events_per_video_min": int(min(counts)) if counts else None,
        "events_per_video_max": int(max(counts)) if counts else None,
        "total_positive_time_s": round(sum(p["positive_time_s"] for p in per), 3),
        "prevalence": round(sum(p["positive_time_s"] for p in known) / sum(p["duration_s"] for p in known), 4)
        if known else None,
        "by_event_type": {k: int(v) for k, v in ev["event_type"].replace("", "(none)").value_counts().items()},
        "by_annotator": {k: int(v) for k, v in ev["annotator"].value_counts().items()},
        "per_video": per,
    }
    if "blind" in df.columns:
        out["non_blind_rows"] = int((df["blind"] == False).sum())  # noqa: E712
    return out


def validate_label_files(paths, schema_path=None, results_csvs=(), runs_dir=None, videos_dirs=()):
    """Read + validate. Returns (normalized DataFrame or None, ValidationReport, durations)."""
    schema = load_schema(schema_path)
    paths = [Path(p) for p in paths]
    rep = ValidationReport()
    missing = [str(p) for p in paths if not p.is_file()]
    if missing or not paths:
        rep.errors.append(f"Label file(s) not found: {missing or '(none given)'}")
        return None, rep, {}
    raw = read_label_files(paths)
    durations = collect_durations(results_csvs, runs_dir, videos_dirs,
                                  float(schema.get("agreement", {}).get("grid_fps", 25.0)))
    rep = validate_labels(raw, schema, durations)
    df = normalize_labels(raw) if rep.form != "unknown" and not raw.empty else None
    return df, rep, durations


# --------------------------------------------------------------------------- training gate

@dataclass
class GateResult:
    ok: bool
    messages: list = field(default_factory=list)   # reasons for refusal
    warnings: list = field(default_factory=list)
    labels: pd.DataFrame | None = None             # rows used for training (one annotator per video)
    report: ValidationReport | None = None
    annotator_by_video: dict = field(default_factory=dict)


def training_gate(label_paths, pipeline_csvs, schema_path=None, videos_dirs=(), adjudicator: str = "adjudicated",
                  min_events: int | None = None, allow_unmarked: bool = False) -> GateResult:
    """Decide whether the learned risk model may be trained. Refuses (ok=False) when labels are
    missing or invalid, when a video has >1 annotator and no adjudicated labels, when a training
    video is not marked completely reviewed, or when there are too few events / labelled videos."""
    schema = load_schema(schema_path)
    paths = expand_label_paths(label_paths)
    if not paths or not all(p.is_file() for p in paths):
        where = ", ".join(str(p) for p in label_paths) if label_paths else "(none given)"
        return GateResult(False, [f"No risk-event label files found at {where}. No surgeon/expert risk or near-miss "
                                  f"labels exist yet, so the learned risk model cannot be trained or evaluated."])
    csvs = [Path(c) for c in pipeline_csvs]
    missing_csv = [str(c) for c in csvs if not c.is_file()]
    if missing_csv:
        return GateResult(False, [f"Pipeline results not found: {missing_csv}"])
    df, rep, _ = validate_label_files(paths, schema_path, csvs, None, videos_dirs)
    res = GateResult(False, report=rep, warnings=list(rep.warnings))
    if not rep.ok:
        res.messages.append(f"Label validation failed with {len(rep.errors)} error(s); fix them "
                            f"(python scripts/validate_risk_labels.py ...):")
        res.messages += [f"  {e}" for e in rep.errors[:30]]
        return res
    if schema.get("status") != "confirmed":
        res.warnings.append("configs/risk_events.yaml status is not 'confirmed': the event definitions are still "
                            "engineering PROPOSALS, not surgeon-confirmed definitions.")

    # One annotator per video: the adjudicated labels where they exist, else the only annotator.
    chosen, unresolved = {}, []
    for vid, g in df.groupby("video_id"):
        anns = sorted(g["annotator"].unique())
        if adjudicator in anns:
            chosen[vid] = adjudicator
        elif len(anns) == 1:
            chosen[vid] = anns[0]
        else:
            unresolved.append(f"{vid} ({', '.join(anns)})")
    if unresolved:
        res.messages.append(f"Videos labelled by several annotators without adjudicated labels (annotator "
                            f"'{adjudicator}'): {unresolved}. Adjudicate them first (docs/RISK_ANNOTATION.md).")
        return res
    labels = pd.concat([df[(df.video_id == v) & (df.annotator == a)] for v, a in chosen.items()], ignore_index=True)

    frames = []
    for c in csvs:
        f = pd.read_csv(c, usecols=["video_id", "timestamp_s"], dtype={"video_id": str})
        frames.append(f.assign(_csv=str(c)))
    frames = pd.concat(frames, ignore_index=True)
    per_csv = frames.groupby("video_id")["_csv"].nunique()
    dups = sorted(per_csv[per_csv > 1].index)
    if dups:
        res.messages.append(f"video_id(s) {dups} appear in more than one pipeline CSV; pass one results.csv per video.")
        return res
    pipe_videos = set(frames["video_id"])
    unused = sorted(set(labels["video_id"]) - pipe_videos)
    if unused:
        res.warnings.append(f"Labelled video(s) without a pipeline CSV are ignored: {unused}")
    labels = labels[labels["video_id"].isin(pipe_videos)].reset_index(drop=True)

    if rep.form == "interval":
        mtype = marker_type(schema)
        for vid, g in labels.groupby("video_id"):
            t = frames.loc[frames.video_id == vid, "timestamp_s"].to_numpy(float)
            marks = list(zip(g.loc[g.event_type == mtype, "start_s"], g.loc[g.event_type == mtype, "end_s"]))
            covered = interval_mask(t, marks).mean() if len(t) else 0.0
            if covered < 1.0:
                msg = (f"video '{vid}' (annotator '{chosen[vid]}'): only {100 * covered:.0f}% of its pipeline frames "
                       f"lie inside a '{mtype}' row, so unreviewed frames would be used as negatives.")
                (res.warnings if allow_unmarked else res.messages).append(msg + (
                    "" if allow_unmarked else " Mark the video reviewed in the annotation tool, or pass "
                                              "--allow-unmarked-videos if the label file is known to be complete."))
        n_events = int(((labels["label"] == 1) & (labels["event_type"] != mtype)).sum())
    else:
        n_events = int((labels["label"] == 1).sum())  # frame form: positive frames
    min_events = int(min_events if min_events is not None else schema.get("training", {}).get("min_events", 10))
    n_vid = labels["video_id"].nunique()
    if n_vid < 2:
        res.messages.append(f"Labels cover {n_vid} video(s) with pipeline results; a video-level train/test split "
                            f"needs >= 2 (in practice many more).")
    if n_events < min_events:
        res.messages.append(f"Too few labelled risk events: {n_events} (need >= {min_events}).")
    if res.messages:
        return res
    res.ok, res.labels, res.annotator_by_video = True, labels, chosen
    return res
