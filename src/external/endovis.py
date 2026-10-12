"""EndoVis'15 robotic-instrument joint annotations (Du et al., IEEE TMI 2018) - validation and DLC conversion.

This is an EXTERNAL, SEPARATE experiment. The instruments are da Vinci robotic claspers/scissors
(ex vivo), not the laparoscopic L-hook / grasper of the CholecSeg8k-based PRISM-Endo model. Joints are
kept under their source names and are NEVER mapped to lhook_tip / grasper_tip.

Source formats (verified by inspection):
  * <split>N_labels.json: list with one entry per video frame; entry["filename"] ends with
    .../DatasetK/Raw/img_<NNNNNN>_raw.png (NNNNNN = 1-based frame number); entry["annotations"] is a list
    of {"class", "id", "x", "y"} with class in LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint,
    EndPoint and an undocumented TrackedPoint, id in tool1..tool4. No visibility flag: an absent joint is
    the only "not visible" signal. Pixel coordinates in the 720x576 frame.
  * DatasetK/(Left_Instrument_|Right_Instrument_)Pose.txt: the ORIGINAL EndoVis'15 tracking annotation
    (one line per frame: x y + axis vectors + angle), a different label type; not used as joint labels.
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

JOINTS = ["LeftClasperPoint", "RightClasperPoint", "HeadPoint", "ShaftPoint", "EndPoint", "TrackedPoint"]
TOOLS = {"tool1": "right_clasper", "tool2": "left_clasper", "tool3": "right_scissor", "tool4": "left_scissor"}
FILE_RE = re.compile(r"Dataset(\d+)/Raw/img_(\d+)(?:_raw)?\.png$")  # train: img_000001_raw.png, test: img_0001.png


@dataclass
class FrameLabels:
    split: str              # train | test (source file group)
    seq: int                # DatasetK
    frame_number: int       # 1-based, from the filename
    points: dict = field(default_factory=dict)   # (tool, joint) -> (x, y)
    issues: list = field(default_factory=list)


def load_labels(json_path: Path, split: str) -> tuple[list[FrameLabels], dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    frames, stats = [], collections.Counter()
    for e in data:
        stats["entries"] += 1
        m = FILE_RE.search(e.get("filename", "").replace("\\", "/"))
        if not m:
            stats["unparsable_filename"] += 1
            continue
        fl = FrameLabels(split, int(m[1]), int(m[2]))
        for a in e.get("annotations", []):
            key = (a.get("id"), a.get("class"))
            x, y = a.get("x"), a.get("y")
            if key[1] not in JOINTS:
                fl.issues.append(f"unknown joint {key[1]}")
            if key[0] not in TOOLS:
                fl.issues.append(f"unknown tool id {key[0]}")
            if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (x, y)):
                fl.issues.append(f"non-numeric coordinate {key}")
                continue
            if key in fl.points:
                fl.issues.append(f"duplicate {key}")
                continue
            fl.points[key] = (float(x), float(y))
        if fl.points:
            stats["labelled_frames"] += 1
        frames.append(fl)
    return frames, dict(stats)


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {"readable": cap.isOpened(), "frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS)}
    n = 0
    while True:  # count decodable frames (container counts can be off)
        ok = cap.grab()
        if not ok:
            break
        n += 1
    info["frames_decoded"] = n
    cap.release()
    return info


def read_frame(video: Path, frame_number: int) -> np.ndarray | None:
    """frame_number is the 1-based number from the annotation filename -> video index frame_number-1."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def validate(frames: list[FrameLabels], videos: dict[int, Path], width: int = 720, height: int = 576) -> dict:
    infos = {k: video_info(v) for k, v in videos.items()}
    rep = collections.Counter()
    per_seq = collections.defaultdict(collections.Counter)
    flagged = []
    for f in frames:
        if not f.points:
            continue
        s = per_seq[f.seq]
        s["labelled"] += 1
        if f.seq not in infos:
            s["no_video"] += 1
            rep["unmatched_no_video"] += 1
            continue
        if f.frame_number < 1 or f.frame_number > infos[f.seq]["frames_decoded"]:
            s["frame_out_of_range"] += 1
            rep["unmatched_frame_out_of_range"] += 1
            continue
        s["matched"] += 1
        rep["matched"] += 1
        for key, (x, y) in f.points.items():
            if not (0 <= x < width and 0 <= y < height):
                f.issues.append(f"outside image {key}: ({x:.1f}, {y:.1f})")
        tools = {t for t, _ in f.points}
        for t in tools:
            missing = [j for j in JOINTS if (t, j) not in f.points]
            if missing:
                f.issues.append(f"{t} missing joints {missing}")
        if f.issues:
            flagged.append({"seq": f.seq, "frame": f.frame_number, "issues": f.issues})
            rep["frames_with_issues"] += 1
    return {"videos": {k: infos[k] for k in sorted(infos)}, "summary": dict(rep),
            "per_sequence": {k: dict(v) for k, v in sorted(per_seq.items())}, "flagged": flagged}


def draw(img: np.ndarray, f: FrameLabels) -> np.ndarray:
    out = img.copy()
    colors = {"tool1": (0, 255, 255), "tool2": (255, 0, 255), "tool3": (0, 255, 0), "tool4": (255, 128, 0)}
    order = ["EndPoint", "TrackedPoint", "ShaftPoint", "HeadPoint"]
    for t in sorted({t for t, _ in f.points}, key=str):
        c = colors.get(t, (255, 255, 255))
        chain = [f.points[(t, j)] for j in order if (t, j) in f.points]
        for a, b in zip(chain, chain[1:]):
            cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), c, 1, cv2.LINE_AA)
        for j in JOINTS:
            if (t, j) in f.points:
                x, y = map(int, f.points[(t, j)])
                cv2.circle(out, (x, y), 4, c, -1, cv2.LINE_AA)
                cv2.putText(out, j.replace("Point", "")[:6], (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1,
                            cv2.LINE_AA)
    cv2.putText(out, f"Dataset{f.seq} frame {f.frame_number} (video index {f.frame_number - 1})", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def bodyparts() -> list[str]:
    """Source-faithful DLC bodypart names, e.g. tool1_LeftClasperPoint (never PRISM tip names)."""
    return [f"{t}_{j}" for t in ("tool1", "tool2") for j in JOINTS]


def to_dlc_table(frames: list[FrameLabels], scorer: str, video_stem: str, n_frames: int):
    """CollectedData table (DLC >= 2.3 row MultiIndex). Absent joints -> NaN (not visible / not labelled)."""
    import math

    import pandas as pd

    bps = bodyparts()
    width = max(1, int(math.ceil(math.log10(n_frames))))
    idx, rows = [], []
    for f in sorted(frames, key=lambda f: f.frame_number):
        if not f.points:
            continue
        idx.append(("labeled-data", video_stem, f"img{str(f.frame_number - 1).zfill(width)}.png"))
        rows.append([v for bp in bps for v in f.points.get(tuple(bp.split("_", 1)), (np.nan, np.nan))])
    cols = pd.MultiIndex.from_product([[scorer], bps, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx), columns=cols).astype(float)
