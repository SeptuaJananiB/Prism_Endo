"""Streaming frame sources (video files or image-sequence folders) and writers.

Frames are yielded one at a time so long videos are never fully loaded into
memory. Every frame carries a timestamp in seconds: for video files it comes
from the container (CAP_PROP_POS_MSEC) when that is valid, otherwise from
frame_index / fps; for image folders it comes from the frame number parsed
from the filename divided by a configured fps (so gaps between clips are
represented as real time gaps rather than silently collapsed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
_FRAME_NUM_RE = re.compile(r"frame_(\d+)")
_ANY_NUM_RE = re.compile(r"(\d+)")


@dataclass
class Frame:
    index: int            # sequential index in this run (0, 1, 2, ...)
    source_index: int     # frame number in the original video / filename
    timestamp_s: float    # seconds since start of source
    image: np.ndarray     # BGR uint8
    name: str = ""


def _frame_number(path: Path) -> int | None:
    m = _FRAME_NUM_RE.search(path.stem) or _ANY_NUM_RE.search(path.stem)
    return int(m.group(1)) if m else None


def list_image_frames(folder: Path, pattern: str | None = None) -> list[Path]:
    """List raw frames in a folder (recursively). For CholecSeg8k folders only
    the `*_endo.png` images are returned, never the mask files."""
    folder = Path(folder)
    if pattern:
        files = sorted(folder.rglob(pattern))
    else:
        endo = sorted(folder.rglob("*_endo.png"))
        if endo:
            files = endo
        else:
            files = sorted(p for p in folder.rglob("*")
                           if p.suffix.lower() in IMAGE_EXTS and "mask" not in p.stem.lower())
    keyed = [(_frame_number(p), p) for p in files]
    if all(k is not None for k, _ in keyed):
        keyed.sort(key=lambda t: (t[0], str(t[1])))
    return [p for _, p in keyed]


class FrameSource:
    """Iterate frames from a video file or an image-sequence folder."""

    def __init__(self, source: str | Path, fps: float | None = None,
                 max_frames: int | None = None, stride: int = 1,
                 image_pattern: str | None = None):
        self.source = Path(source)
        self.max_frames = max_frames
        self.stride = max(1, int(stride))
        self._cap = None
        self._files: list[Path] = []

        if self.source.is_dir():
            self.kind = "images"
            self._files = list_image_frames(self.source, image_pattern)
            if not self._files:
                raise FileNotFoundError(f"No image frames found in folder: {self.source}")
            self.fps = float(fps or 25.0)
            first = cv2.imread(str(self._files[0]))
            if first is None:
                raise IOError(f"Could not read image: {self._files[0]}")
            self.height, self.width = first.shape[:2]
            self.total_source_frames = len(self._files)
        elif self.source.is_file():
            self.kind = "video"
            cap = cv2.VideoCapture(str(self.source))
            if not cap.isOpened():
                raise IOError(f"Could not open video: {self.source}")
            container_fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps or (container_fps if container_fps and container_fps > 1 else 25.0))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
        else:
            raise FileNotFoundError(f"Input not found: {self.source}")

    def __len__(self) -> int:
        n = -(-self.total_source_frames // self.stride) if self.total_source_frames else 0
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self) -> Iterator[Frame]:
        if self.kind == "images":
            yield from self._iter_images()
        else:
            yield from self._iter_video()

    def _iter_images(self) -> Iterator[Frame]:
        out_idx = 0
        first_num = _frame_number(self._files[0])
        for i, path in enumerate(self._files[::self.stride]):
            if self.max_frames is not None and out_idx >= self.max_frames:
                break
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Could not read image: {path}")
            num = _frame_number(path)
            src_idx = num if num is not None else i * self.stride
            base = first_num if (num is not None and first_num is not None) else 0
            yield Frame(out_idx, src_idx, (src_idx - base) / self.fps, img, path.name)
            out_idx += 1

    def _iter_video(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.source))
        self._cap = cap
        try:
            src_idx, out_idx, last_ts = 0, 0, -1.0
            while True:
                if self.max_frames is not None and out_idx >= self.max_frames:
                    break
                ok, img = cap.read()
                if not ok:
                    break
                if src_idx % self.stride == 0:
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    # Some containers report 0 or non-monotonic times; fall back.
                    if not np.isfinite(ts) or ts <= last_ts or (ts == 0 and src_idx > 0):
                        ts = src_idx / self.fps
                    last_ts = ts
                    yield Frame(out_idx, src_idx, ts, img, f"frame_{src_idx:06d}")
                    out_idx += 1
                src_idx += 1
        finally:
            cap.release()
            self._cap = None


class VideoWriter:
    """Lazily-opened video writer. Picks a codec by file extension and falls
    back through alternatives because codec availability differs per OpenCV
    build (e.g. H.264 is often missing on Windows wheels)."""

    # mp4v first: OpenCV's Windows wheels report avc1 as "opened" even when the
    # OpenH264 DLL is missing. Use to_browser_mp4() for H.264 playback.
    CODECS = {".mp4": ["mp4v", "avc1"], ".webm": ["VP80", "VP90"],
              ".avi": ["MJPG", "XVID"]}

    def __init__(self, path: str | Path, fps: float):
        self.path = Path(path)
        self.fps = float(fps)
        self._writer = None
        self.codec = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._open(frame.shape[1], frame.shape[0])
        self._writer.write(frame)

    def _open(self, w: int, h: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for codec in self.CODECS.get(self.path.suffix.lower(), ["mp4v"]):
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if writer.isOpened():
                self._writer, self.codec = writer, codec
                return
            writer.release()
        raise IOError(f"Could not open a video writer for {self.path} "
                      f"(tried codecs {self.CODECS.get(self.path.suffix.lower())})")

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def ffmpeg_exe() -> str | None:
    """Path to an ffmpeg binary (imageio-ffmpeg's bundled one, or PATH)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil

        return shutil.which("ffmpeg")


def to_browser_mp4(src: str | Path, dst: str | Path | None = None) -> Path | None:
    """Transcode to H.264/yuv420p MP4 so browsers (and Streamlit) can play it.
    Returns the new path, or None if no ffmpeg is available or it fails."""
    import subprocess

    exe = ffmpeg_exe()
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + "_h264.mp4")
    if exe is None or not src.is_file():
        return None
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    except (subprocess.SubprocessError, OSError):
        return None
    return dst if dst.is_file() and dst.stat().st_size > 0 else None
