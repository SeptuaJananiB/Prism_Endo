"""Run a trained segmentation checkpoint on a video, an image folder, or one image.

Saves label-map PNGs (pixel value = class index), colour overlays, and
optionally an overlay video.

  python predict_segmentation.py --checkpoint outputs/models/instrument_seg/best.pt \
      --input video01/video01_14859 --out outputs/predictions/instrument_14859 --video
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from src.segmentation.predictor import SegmentationPredictor, overlay
from src.utils.config import resolve_path
from src.utils.device import select_device
from src.utils.video_io import FrameSource, VideoWriter


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="video file, image folder, or single image")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=None, help="fps for image folders (default 25)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--video", action="store_true", help="also write an overlay video (overlay.mp4)")
    p.add_argument("--no-masks", action="store_true", help="skip writing per-frame label PNGs")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    pred = SegmentationPredictor.from_checkpoint(resolve_path(args.checkpoint), select_device(args.device))
    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    if inp.is_file() and inp.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        img = cv2.imread(str(inp))
        lab = pred.predict(img)
        cv2.imwrite(str(out / "masks" / f"{inp.stem}_label.png"), lab)
        cv2.imwrite(str(out / "overlays" / f"{inp.stem}_overlay.jpg"), overlay(img, lab))
        print(f"Saved prediction for {inp}")
        return 0

    src = FrameSource(inp, fps=args.fps, max_frames=args.max_frames)
    writer = VideoWriter(out / "overlay.mp4", src.fps) if args.video else None
    counts = {n: 0 for n in pred.class_names[1:]}
    n = 0
    try:
        for fr in src:
            lab = pred.predict(fr.image)
            vis = overlay(fr.image, lab)
            stem = Path(fr.name).stem or f"frame_{fr.source_index:06d}"
            if not args.no_masks:
                cv2.imwrite(str(out / "masks" / f"{stem}_label.png"), lab)
            cv2.imwrite(str(out / "overlays" / f"{stem}_overlay.jpg"), vis)
            if writer:
                writer.write(vis)
            for i, name in enumerate(pred.class_names[1:], start=1):
                counts[name] += int((lab == i).any())
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(src)} frames")
    finally:
        if writer:
            writer.release()
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"frames": n, "frames_with_class": counts, "classes": pred.class_names}, f, indent=2)
    print(f"Processed {n} frames -> {out}  (frames containing each class: {counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
