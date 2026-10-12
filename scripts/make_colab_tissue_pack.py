"""Pack CholecSeg8k (448x256, class-index labels via the training LUT) + code + checkpoint for the Colab T4 run of
train_tissue_dsad.py. Frames are subsampled (train/val every 4th, test every 2nd: consecutive frames are near-duplicates)
and images stored as JPEG q95 to fit the 10 MB-per-file upload. Both models are tested on the SAME packed test set.
Output: outputs/cloud/tissue/pack.zip split into tissue_pack.zip.NN parts.

  python scripts/make_colab_tissue_pack.py
"""
import shutil
import sys
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.segmentation.train import make_datasets  # noqa: E402
from src.utils.config import load_config  # noqa: E402

OUT = ROOT / "outputs/cloud/tissue"
STEP = {"train": 4, "val": 4, "test": 2}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    stage = OUT / "stage"
    cfg = load_config(ROOT / "configs/tissue_seg_multivideo.yaml")
    cfg.setdefault("train", {})["cache"] = False
    ds, _, _ = make_datasets(cfg, ("train", "val", "test"))
    for split, d in ds.items():
        od = stage / "cholec_png" / split
        od.mkdir(parents=True)
        for i in range(0, len(d), STEP[split]):
            img, lab = d.load(i)
            s = d.samples[i]
            stem = f"{s.video}_{Path(s.image).stem}"
            cv2.imwrite(str(od / f"{stem}_img.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(od / f"{stem}_lab.png"), lab)
        print(split, len(list(od.glob("*_lab.png"))), flush=True)
    for f in ["scripts/train_tissue_dsad.py", "scripts/prepare_dsad.py", "configs/tissue_seg.yaml",
              "configs/tissue_seg_multivideo.yaml", "outputs/models/tissue_seg_multivideo/best.pt",
              "outputs/models/tissue_seg_dsad/best.pt", "outputs/models/tissue_seg_dsad/history.json"]:
        (stage / f).parent.mkdir(parents=True, exist_ok=True)
        if f.endswith(".pt"):  # fp16 weights halve the upload; load_state_dict casts back to fp32
            import torch
            ck = torch.load(str(ROOT / f), map_location="cpu", weights_only=True)
            ck["model_state"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["model_state"].items()}
            torch.save(ck, stage / f)
        else:
            shutil.copy2(ROOT / f, stage / f)
    for f in (ROOT / "src").rglob("*.py"):
        t = stage / f.relative_to(ROOT)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
    z = OUT / "pack.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in stage.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(stage))
    shutil.rmtree(stage)
    data, n = z.read_bytes(), 9_500_000
    for k in range(0, len(data), n):
        (OUT / f"tissue_pack.zip.{k // n:02d}").write_bytes(data[k:k + n])
    z.unlink()
    print("size MB", round(len(data) / 1e6, 1), "parts", -(-len(data) // n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
