"""Synthetic test clip with a KNOWN instrument tip trajectory (smoke tests only).

Draws an endoscope-like circular field on black borders, a red "tissue"
background, a yellow "target" region, and a grey shaft entering from the
left edge whose tip follows a known path towards the target. Writes the
video plus `<name>_tip_truth.csv` (frame, x, y) and `<name>_target_mask.png`.

This is a code-path fixture, NOT surgical data: results on it say nothing
about real performance.

  python scripts/make_synthetic_video.py --out outputs/synthetic/synthetic.mp4
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.video_io import VideoWriter  # noqa: E402


def render(n_frames=75, w=854, h=480, fps=25.0):
    rng = np.random.default_rng(0)
    target = np.zeros((h, w), np.uint8)
    cv2.ellipse(target, (620, 260), (90, 60), 0, 0, 360, 1, -1)
    frames, tips = [], []
    for i in range(n_frames):
        s = i / (n_frames - 1)
        # Tip moves right towards the target, slows down, then retreats a little.
        tip_x = 300 + 260 * np.sin(s * np.pi * 0.8)
        tip_y = 300 - 40 * s
        img = np.zeros((h, w, 3), np.uint8)
        fov = np.zeros((h, w), np.uint8)
        cv2.circle(fov, (w // 2, h // 2), 330, 1, -1)
        tissue = np.clip(np.array([70, 80, 170]) + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)
        img[fov > 0] = tissue[fov > 0]
        img[(target > 0) & (fov > 0)] = (60, 200, 220)
        entry = (w // 2 - 320, int(tip_y + 60))
        cv2.line(img, entry, (int(tip_x), int(tip_y)), (150, 150, 150), 16, cv2.LINE_AA)
        img[fov == 0] = 0
        frames.append(img)
        tips.append((i, float(tip_x), float(tip_y)))
    return frames, tips, target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/synthetic/synthetic.mp4")
    p.add_argument("--frames", type=int, default=75)
    args = p.parse_args(argv)
    out = Path(args.out)
    frames, tips, target = render(args.frames)
    with VideoWriter(out, 25.0) as w:
        for f in frames:
            w.write(f)
    with open(out.with_name(out.stem + "_tip_truth.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["frame", "x", "y"])
        cw.writerows(tips)
    cv2.imwrite(str(out.with_name(out.stem + "_target_mask.png")), target * 255)
    print(f"wrote {out} ({len(frames)} frames) + tip truth + target mask")
    return 0


if __name__ == "__main__":
    sys.exit(main())
