"""Thin, documented wrappers around the DeepLabCut 3.x API (PyTorch engine).

DLC is imported lazily so the rest of PRISM-Endo works without it. Tested
against deeplabcut==3.0.2 on Python 3.11 (CPU). The labelling GUI is a
separate optional install (`pip install "deeplabcut[gui]"`).

Workflow
--------
1. create_project      - new single-animal project whose "bodyparts" are the
                         configured instrument keypoints (e.g. lhook_tip).
2. extract_frames      - k-means selection of visually diverse frames.
3. (human) label        - label frames in the DLC/napari GUI. Leave a point
                         unlabelled when it is not visible; unlabelled points
                         are NaN and are never used as ground truth.
4. build_dataset        - create_training_dataset (train/test split by DLC).
5. train                - train_network (pretrained ImageNet backbone).
6. evaluate             - evaluate_network: train/test pixel error.
7. analyze              - analyze_videos -> <video>DLC_<scorer>.h5/.csv, the
                         file the PRISM-Endo pipeline reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _dlc():
    try:
        import deeplabcut  # noqa: F401
        return deeplabcut
    except ImportError as e:
        raise ImportError(
            "DeepLabCut is not installed in this environment. Use the project venv "
            "(.venv) or install with: pip install \"deeplabcut==3.0.2\" (add [gui] for labelling).") from e


def create_project(project: str, experimenter: str, videos: list[str | Path], working_dir: str | Path,
                   bodyparts: list[str], numframes2pick: int = 20) -> Path:
    dlc = _dlc()
    from deeplabcut.utils.auxiliaryfunctions import edit_config

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = dlc.create_new_project(project, experimenter, [str(v) for v in videos],
                                      working_directory=str(working_dir), copy_videos=True)
    if cfg_path == "nothingcreated" or cfg_path is None:
        raise RuntimeError("DeepLabCut did not create a project (does it already exist?)")
    edit_config(str(cfg_path), {"bodyparts": list(bodyparts), "skeleton": [],
                                "numframes2pick": int(numframes2pick), "engine": "pytorch"})
    return Path(cfg_path)


def add_videos(config: str | Path, videos: list[str | Path]) -> None:
    _dlc().add_new_videos(str(config), [str(v) for v in videos], copy_videos=True)


def extract_frames(config: str | Path, algo: str = "kmeans") -> None:
    _dlc().extract_frames(str(config), mode="automatic", algo=algo, userfeedback=False)


def label_frames(config: str | Path) -> None:
    dlc = _dlc()
    if getattr(dlc, "label_frames", None) is None:
        raise ImportError(
            "The DLC labelling GUI is not installed. Install it (ideally in a separate env):\n"
            "  pip install \"deeplabcut[gui]==3.0.2\"\n"
            f"then run: python -c \"import deeplabcut; deeplabcut.label_frames(r'{config}')\"")
    dlc.label_frames(str(config))


def build_dataset(config: str | Path, net_type: str = "resnet_50") -> None:
    dlc = _dlc()
    from deeplabcut.core.engine import Engine

    dlc.create_training_dataset(str(config), net_type=net_type, engine=Engine.PYTORCH, userfeedback=False)


def train(config: str | Path, epochs: int = 200, batch_size: int = 8, device: str = "auto",
          save_epochs: int = 25, shuffle: int = 1) -> None:
    dlc = _dlc()
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dlc.train_network(str(config), shuffle=shuffle, epochs=epochs, save_epochs=save_epochs,
                      batch_size=batch_size, device=device)


def evaluate(config: str | Path, shuffle: int = 1, plotting: bool = True) -> None:
    _dlc().evaluate_network(str(config), shuffles=[shuffle], plotting=plotting)


def analyze(config: str | Path, video: str | Path, out_dir: str | Path, device: str = "auto",
            shuffle: int = 1, labeled_video: bool = True) -> Path:
    """Run a trained DLC model on a video; returns the prediction .h5 path."""
    dlc = _dlc()
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dlc.analyze_videos(str(config), [str(video)], shuffle=shuffle, save_as_csv=True,
                       destfolder=str(out_dir), device=device)
    if labeled_video:
        dlc.create_labeled_video(str(config), [str(video)], shuffle=shuffle, destfolder=str(out_dir))
    hits = sorted(out_dir.glob(f"{Path(video).stem}DLC*.h5"))
    hits = [h for h in hits if "_meta" not in h.stem]
    if not hits:
        raise FileNotFoundError(f"DLC analysis finished but no prediction file found in {out_dir}")
    return hits[0]


def label_status(config: str | Path) -> dict:
    """Count HUMAN labels (CollectedData_*.h5/.csv) per video and bodypart.
    NaN coordinates are unlabelled and not counted. Works without importing DLC."""
    import yaml

    from src.tracking.dlc_io import read_dlc_table, table_to_keypoints

    config = Path(config)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    bodyparts = list(cfg.get("bodyparts") or [])
    root = config.parent / "labeled-data"
    status = {"bodyparts": bodyparts, "videos": {}, "frames_extracted": 0, "frames_labelled": 0,
              "per_bodypart": {b: 0 for b in bodyparts}}
    for vid in sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []:
        imgs = list(vid.glob("img*.png"))
        status["frames_extracted"] += len(imgs)
        files = sorted(vid.glob("CollectedData_*.h5")) or sorted(vid.glob("CollectedData_*.csv"))
        row = {"extracted": len(imgs), "labelled_frames": 0, "per_bodypart": {b: 0 for b in bodyparts}}
        if files:
            kp = table_to_keypoints(read_dlc_table(files[0]))
            any_lab = None
            for b, d in kp.items():
                lab = np.isfinite(d["x"]) & np.isfinite(d["y"])
                row["per_bodypart"][b] = int(lab.sum())
                any_lab = lab if any_lab is None else (any_lab | lab)
            row["labelled_frames"] = int(any_lab.sum()) if any_lab is not None else 0
        status["videos"][vid.name] = row
        status["frames_labelled"] += row["labelled_frames"]
        for b, c in row["per_bodypart"].items():
            status["per_bodypart"][b] = status["per_bodypart"].get(b, 0) + c
    return status


def require_labels(config: str | Path, min_frames: int = 100, min_per_bodypart: int = 30,
                   optional_suffix: str = "_2") -> dict:
    """Raise unless enough human-labelled frames exist (prevents training on nothing).
    Extra-instance bodyparts (e.g. grasper_tip_2) are optional: in the current train clips only
    13 frames show two graspers, so their minimum is reported as a warning, not enforced."""
    st = label_status(config)
    short = {b: c for b, c in st["per_bodypart"].items() if c < min_per_bodypart and not b.endswith(optional_suffix)}
    st["warnings"] = [f"{b}: only {c} labels (optional bodypart; the model will rarely predict it)"
                      for b, c in st["per_bodypart"].items() if c < min_per_bodypart and b.endswith(optional_suffix)]
    if st["frames_labelled"] < min_frames or short:
        raise RuntimeError(
            f"Not enough human labels to train: {st['frames_labelled']} labelled frames "
            f"(need >= {min_frames}); bodyparts below {min_per_bodypart} labels: {short or 'none'}. "
            f"Label the frames in the DLC GUI first (see docs/DLC_ANNOTATION.md).")
    return st
