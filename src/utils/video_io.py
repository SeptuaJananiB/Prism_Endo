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


def live_target(source) -> int | str | None:
    """Live capture source: 'camera:N' / 'N' (webcam or capture card) -> int, 'rtsp://...' / 'http(s)://...' -> URL,
    'video:<path>' -> a video file replayed at its own frame rate through the live pipeline (no camera in between)."""
    s = str(source).strip()
    if s.lower().startswith("camera:") and s[7:].isdigit():
        return int(s[7:])
    if s.isdigit():
        return int(s)
    if s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        return s
    if s.lower().startswith("video:") and len(s) > 6:  # a video file played in real time as if it were a camera
        return s
    return None


class FrameSource:
    """Iterate frames from a video file, an image-sequence folder, or a LIVE source (camera index / stream URL)."""

    def __init__(self, source: str | Path, fps: float | None = None,
                 max_frames: int | None = None, stride: int = 1,
                 image_pattern: str | None = None, live_max_width: int | None = 854):
        self.live = live_target(source)
        self.source = Path(str(source)) if self.live is None else str(source)
        self.max_frames = max_frames
        self.stride = max(1, int(stride))
        self._cap = None
        self._files: list[Path] = []

        if self.live is not None:
            self.kind = "live"
            cap = _open_live(self.live)
            ok, first = cap.read()
            if not ok or first is None:
                cap.release()
                raise IOError(f"No frames from live source {source} (camera busy, wrong index or URL?)")
            self._cap = cap
            self._first = first
            self.live_max_width = int(live_max_width) if live_max_width else 0
            self.on_capture = None  # set by the caller: called with every captured (downscaled) frame
            reported = cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps or (reported if reported and 1 < reported <= 120 else 30.0))
            self.height, self.width = self._small(first).shape[:2]
            self.camera_size = (first.shape[1], first.shape[0])
            self.total_source_frames = 0  # unknown: runs until stopped (or max_frames)
        elif self.source.is_dir():
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
        if self.kind == "live":
            yield from self._iter_live()
            return
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

    def _iter_live(self) -> Iterator[Frame]:
        """Real-time capture. A background thread reads the camera at its own frame rate (30/60 fps), keeps only the
        NEWEST frame for analysis (older ones are dropped, so a slower analysis never lags behind) and hands EVERY
        frame to `on_capture` for a smooth live display. Frames are downscaled to `live_max_width` (default 854 px,
        the resolution the models were trained at) - large webcam frames otherwise slow every stage down."""
        import threading
        import time

        cap, t0 = self._cap, time.perf_counter()
        lock, fresh, stop = threading.Lock(), threading.Event(), threading.Event()
        latest: dict = {"img": self._small(self._first), "n": 0}
        fresh.set()

        def reader():
            n = 0
            while not stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                small = self._small(img)
                with lock:
                    latest["img"], latest["n"] = small, n
                fresh.set()
                cb = self.on_capture
                if cb is not None:
                    try:
                        cb(small)
                    except Exception:  # display problems must never stop the capture
                        pass
            stop.set()
            fresh.set()

        th = threading.Thread(target=reader, daemon=True, name="live-capture")
        th.start()
        out_idx = 0
        try:
            while not stop.is_set() or fresh.is_set():
                if self.max_frames is not None and out_idx >= self.max_frames:
                    break
                fresh.wait(timeout=2.0)
                with lock:
                    img, n = latest["img"], latest["n"]
                    latest["img"] = None
                    fresh.clear()
                if img is None:
                    if stop.is_set():
                        break
                    continue
                yield Frame(out_idx, n, time.perf_counter() - t0, img, f"live_{n:06d}")
                out_idx += 1
        finally:
            stop.set()
            th.join(timeout=2.0)
            cap.release()

    def _small(self, img):
        w = img.shape[1]
        if self.live_max_width and w > self.live_max_width:
            h = int(round(img.shape[0] * self.live_max_width / w))
            return cv2.resize(img, (self.live_max_width, h), interpolation=cv2.INTER_AREA)
        return img

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


class _PacedVideo:
    """A video file that behaves like a camera: read() returns frames at the file's own frame rate (real time),
    so the live pipeline sees the original pixels with no webcam, screen glare or colour shift in between."""

    def __init__(self, path: str):
        import time

        self._time = time
        self.cap = cv2.VideoCapture(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.dt = 1.0 / (fps if 1 <= fps <= 120 else 25.0)
        self.next = None

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        now = self._time.perf_counter()
        if self.next is None:
            self.next = now
        if self.next > now:
            self._time.sleep(self.next - now)
        self.next = max(self.next + self.dt, self._time.perf_counter() - self.dt)  # never run ahead, never pile up
        return self.cap.read()

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, *_a):
        return True

    def release(self):
        self.cap.release()


def _open_live(target: int | str):
    """Open a webcam / capture card (DirectShow on Windows, small buffer) or a network stream with OpenCV."""
    import sys

    if isinstance(target, str) and target.lower().startswith("video:"):
        cap = _PacedVideo(target[6:])
        if not cap.isOpened():
            raise IOError(f"Could not open video {target[6:]}")
        return cap
    if isinstance(target, int):
        cap = cv2.VideoCapture(target, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(target)
    else:
        cap = cv2.VideoCapture(target)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(target, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG unlocks 30/60 fps on most webcams
        cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise IOError(f"Could not open live source {target}")
    return cap


def list_cameras(max_index: int = 4) -> list[int]:
    """Indexes of cameras / capture cards that deliver a frame (quick probe)."""
    found = []
    for i in range(max_index):
        try:
            cap = _open_live(i)
            ok, _ = cap.read()
            cap.release()
            if ok:
                found.append(i)
        except IOError:
            pass
    return found
