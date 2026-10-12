"""Compare two tissue checkpoints on frames of a video: fraction of frames / pixels called each tissue, plus a side-by-side
image grid. Used to check false "liver" on non-cholecystectomy video (no ground truth: descriptive only).

  python scripts/compare_tissue_on_video.py --video Videos/appendicectomy_max.mp4 --n 40
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.segmentation.predictor import SegmentationPredictor, overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--a", default="outputs/models/tissue_seg_multivideo/best.pt")
    ap.add_argument("--b", default="outputs/models/tissue_seg_dsad_colab/best.pt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="outputs/tissue_compare")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / a.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, a.n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append((int(i), f))
    res, tiles = {}, []
    for tag, ck in (("original", a.a), ("dsad_finetuned", a.b)):
        P = SegmentationPredictor.from_checkpoint(ROOT / ck)
        names = P.class_names
        pix, frm, ov = np.zeros(len(names)), np.zeros(len(names)), []
        for i, f in frames:
            lab = P.predict(f)
            c = np.bincount(lab.ravel(), minlength=len(names))[:len(names)] / lab.size
            pix += c
            frm += c > 0.02
            ov.append(cv2.resize(overlay(f, lab), (320, 180)))
        res[tag] = {n: {"mean_pixel_frac": round(float(p / len(frames)), 4), "frames_over_2pct": int(q)}
                    for n, p, q in zip(names, pix, frm)}
        tiles.append(ov)
    rows = [np.hstack([tiles[0][k], tiles[1][k]]) for k in range(0, len(frames), max(1, len(frames) // 8))][:8]
    cv2.imwrite(str(out / "original_vs_dsad.jpg"), np.vstack(rows))
    rep = {"video": a.video, "frames_sampled": len(frames), "note": "no ground truth; left = original, right = DSAD",
           **res}
    (out / "compare.json").write_text(json.dumps(rep, indent=2))
    for n in ("liver", "gastrointestinal_tract", "abdominal_wall", "fat", "gallbladder"):
        print(f"{n:24s} original {res['original'][n]} | dsad {res['dsad_finetuned'][n]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
