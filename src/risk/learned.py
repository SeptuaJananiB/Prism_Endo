"""Mode 2: learned risk model - only for when expert risk/event labels exist.

No risk labels exist in this repository. This module therefore refuses to
train unless a labels file is supplied and minimum data requirements are
met. It never generates labels.

Label file (CSV), either form:
  interval form : video_id,start_s,end_s,label          (label 1 = risk event; frames inside -> 1)
                  [,event_type,severity,annotator,confidence,notes,blind,created_at]  (optional)
  frame form    : video_id,source_index,label            (one row per labelled frame)
Frames of a labelled video not covered by an interval are 0 (negative) in
interval form. `video_id` must match the `video_id` column the pipeline
writes into its per-frame CSV (the input file/folder stem). Label-0 rows (e.g.
the `review_complete` coverage marker) never overwrite label-1 intervals.
Create, validate and adjudicate labels with the workflow in
docs/RISK_ANNOTATION.md (src/risk/labels.py, scripts/validate_risk_labels.py,
src/ui/annotate_events.py); scripts/train_risk_model.py refuses to train
unless those checks pass.

Features are computed from the pipeline's per-frame CSV with the SAME code
used online (`frame_base_features` + `window_features`), so training and
inference cannot drift apart.

Splits are by video (GroupShuffleSplit / explicit lists): frames of one
video never appear in both train and test.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = ["min_distance_norm", "inside_any", "max_speed_norm", "max_approach_norm",
                 "max_accel_norm", "max_turn_deg", "n_valid_keypoints"]


def keypoint_names_from_columns(cols) -> list[str]:
    return sorted({c[: -len("_kp_status")] for c in cols if c.endswith("_kp_status")})


def frame_base_features(row: dict | pd.Series, kps: list[str]) -> dict:
    diag = float(row.get("image_diag", np.nan))

    def num(k):
        v = row.get(k, np.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    dist, speed, appr, acc, turn, inside, nvalid = [], [], [], [], [], [], 0
    for kp in kps:
        if str(row.get(f"{kp}_kp_status", "")) == "ok":
            nvalid += 1
        dist.append(num(f"{kp}_prox_distance_norm"))
        speed.append(num(f"{kp}_speed_px_s") / diag)
        appr.append(num(f"{kp}_prox_approach_speed_px_s") / diag)
        acc.append(num(f"{kp}_accel_px_s2") / diag)
        turn.append(num(f"{kp}_turn_angle_deg"))
        ins = row.get(f"{kp}_prox_inside")
        inside.append(1.0 if str(ins) in ("True", "1", "1.0") else 0.0)

    def agg(fn, vals):
        vals = [v for v in vals if np.isfinite(v)]
        return fn(vals) if vals else np.nan

    return {"min_distance_norm": agg(min, dist), "inside_any": max(inside) if inside else 0.0,
            "max_speed_norm": agg(max, speed), "max_approach_norm": agg(max, appr),
            "max_accel_norm": agg(max, acc), "max_turn_deg": agg(max, turn),
            "n_valid_keypoints": float(nvalid)}


def window_features(base: pd.DataFrame, window: int) -> pd.DataFrame:
    """Causal rolling statistics (only past and current frames)."""
    out = base.copy()
    r = base.rolling(window, min_periods=1)
    for c in ("min_distance_norm",):
        out[f"{c}_wmin"] = r[c].min()
    for c in ("max_speed_norm", "max_approach_norm", "max_accel_norm"):
        out[f"{c}_wmean"] = r[c].mean()
        out[f"{c}_wmax"] = r[c].max()
    out["distance_trend"] = base["min_distance_norm"] - base["min_distance_norm"].shift(window - 1)
    return out


def features_from_pipeline_csv(path: str | Path, window: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    kps = keypoint_names_from_columns(df.columns)
    base = pd.DataFrame([frame_base_features(r, kps) for _, r in df.iterrows()])
    feats = window_features(base, window)
    feats.insert(0, "video_id", df["video_id"].astype(str).values)
    feats.insert(1, "source_index", df["source_index"].values)
    feats.insert(2, "timestamp_s", df["timestamp_s"].values)
    return feats


def attach_labels(feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    feats["label"] = np.nan
    if {"start_s", "end_s"} <= set(labels.columns):
        for vid in labels["video_id"].astype(str).unique():
            m = feats["video_id"] == vid
            feats.loc[m, "label"] = 0
            rows = labels[labels["video_id"].astype(str) == vid]
            # Label-0 rows (coverage markers) first so they never erase a label-1 interval.
            rows = rows.assign(_lab=rows["label"].astype(float)).sort_values("_lab", kind="stable")
            for _, ev in rows.iterrows():
                inside = m & (feats["timestamp_s"] >= float(ev["start_s"])) & (feats["timestamp_s"] <= float(ev["end_s"]))
                feats.loc[inside, "label"] = int(ev["label"])
    elif "source_index" in labels.columns:
        key = labels.assign(video_id=labels["video_id"].astype(str)).set_index(["video_id", "source_index"])["label"]
        idx = pd.MultiIndex.from_arrays([feats["video_id"], feats["source_index"]])
        feats["label"] = key.reindex(idx).to_numpy()
    else:
        raise ValueError("Label CSV needs columns video_id,start_s,end_s,label or video_id,source_index,label")
    return feats.dropna(subset=["label"]).assign(label=lambda d: d["label"].astype(int))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("video_id", "source_index", "timestamp_s", "label")]


def train_learned_risk(data: pd.DataFrame, out_dir: str | Path, test_videos: list[str] | None = None,
                       test_fraction: float = 0.3, min_positive_train: int = 20,
                       min_positive_test: int = 5, seed: int = 0, window: int = 25) -> dict:
    """Train a class-balanced logistic regression on labelled frames and
    evaluate on held-out VIDEOS. Raises ValueError if requirements fail."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    videos = sorted(data["video_id"].unique())
    if len(videos) < 2:
        raise ValueError(f"Need labelled frames from >= 2 videos for a video-level split; got {videos}")
    if test_videos:
        test_mask = data["video_id"].isin(test_videos)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        _, te = next(gss.split(data, data["label"], data["video_id"]))
        test_mask = np.zeros(len(data), bool)
        test_mask[te] = True
    train, test = data[~test_mask], data[test_mask]
    pos_tr, pos_te = int(train["label"].sum()), int(test["label"].sum())
    if pos_tr < min_positive_train or pos_te < min_positive_test or (len(test) - pos_te) == 0:
        raise ValueError(f"Not enough labelled events: train positives={pos_tr} (need {min_positive_train}), "
                         f"test positives={pos_te} (need {min_positive_test}), test negatives={len(test) - pos_te}.")
    cols = feature_columns(data)
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(train[cols], train["label"])
    prob = model.predict_proba(test[cols])[:, 1]
    pred = (prob >= 0.5).astype(int)
    y = test["label"].to_numpy()
    metrics = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None,
        "auprc": float(average_precision_score(y, prob)) if len(set(y)) == 2 else None,
        "test_prevalence": float(y.mean()),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "risk_model.joblib")
    # Held-out per-frame probabilities (for event-level warning evaluation).
    pd.DataFrame({"video_id": test["video_id"].to_numpy(), "source_index": test["source_index"].to_numpy(),
                  "timestamp_s": test["timestamp_s"].to_numpy(), "label": y, "prob": prob})         .to_csv(out_dir / "test_predictions.csv", index=False)
    meta = {"feature_columns": cols, "window": window, "train_videos": sorted(train["video_id"].unique()),
            "test_videos": sorted(test["video_id"].unique()), "train_frames": len(train),
            "train_positives": pos_tr, "test_frames": len(test), "test_positives": pos_te,
            "metrics_test": metrics, "model": "LogisticRegression(class_weight=balanced)",
            "test_predictions_csv": str(out_dir / "test_predictions.csv"),
            "note": "Output is P(label=1 | features) under the label definition used; it is only as "
                    "meaningful as the expert labels and is not a validated injury probability."}
    with open(out_dir / "risk_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class LearnedRiskScorer:
    """Online inference with a model produced by train_learned_risk."""

    def __init__(self, model_dir: str | Path):
        import joblib

        model_dir = Path(model_dir)
        if not (model_dir / "risk_model.joblib").is_file():
            raise FileNotFoundError(f"No learned risk model in {model_dir}. Train one with "
                                    f"scripts/train_risk_model.py (requires expert risk labels).")
        # joblib unpickles: only load models you trained yourself.
        self.model = joblib.load(model_dir / "risk_model.joblib")
        with open(model_dir / "risk_model_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.window = int(self.meta["window"])
        self.cols = self.meta["feature_columns"]
        self.buf: deque = deque(maxlen=self.window)

    def update(self, row: dict, kps: list[str]) -> float:
        self.buf.append(frame_base_features(row, kps))
        feats = window_features(pd.DataFrame(list(self.buf)), self.window).iloc[[-1]]
        return float(self.model.predict_proba(feats[self.cols])[0, 1])
