"""Assemble image frames (e.g. a CholecSeg8k clip folder) into an MP4.

DeepLabCut and most video tools need real video files. CholecSeg8k clips are
80 consecutive frames sampled at 25 fps from Cholec80.

  python scripts/make_video_from_frames.py --input video01/video01_14859 --out Videos/video01_14859.mp4
  python scripts/make_video_from_frames.py --all video01 --out-dir Videos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.video_io import FrameSource, VideoWriter, to_browser_mp4  # noqa: E402


def convert(inp: Path, out: Path, fps: float, h264: bool) -> Path:
    src = FrameSource(inp, fps=fps)
    tmp = out.with_name(out.stem + "_mp4v.mp4") if h264 else out
    with VideoWriter(tmp, fps) as w:
        n = 0
        for fr in src:
            w.write(fr.image)
            n += 1
    if h264:
        res = to_browser_mp4(tmp, out)
        if res is None:
            tmp.replace(out)
            print("  ffmpeg unavailable: wrote mp4v (may not play in browsers)")
        else:
            tmp.unlink()
    print(f"{inp} -> {out} ({n} frames @ {fps} fps)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="one folder of frames")
    p.add_argument("--out", help="output .mp4 for --input")
    p.add_argument("--all", help="parent folder: convert every sub-folder that contains frames")
    p.add_argument("--out-dir", default="Videos")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--no-h264", action="store_true", help="skip ffmpeg H.264 transcoding")
    args = p.parse_args(argv)
    if args.input:
        convert(Path(args.input), Path(args.out or Path(args.out_dir) / f"{Path(args.input).name}.mp4"),
                args.fps, not args.no_h264)
    elif args.all:
        for d in sorted(p for p in Path(args.all).iterdir() if p.is_dir()):
            convert(d, Path(args.out_dir) / f"{d.name}.mp4", args.fps, not args.no_h264)
    else:
        p.error("give --input or --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
