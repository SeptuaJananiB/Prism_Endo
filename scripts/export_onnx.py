"""Export segmentation checkpoints to ONNX (model.onnx next to each checkpoint) and verify
that ONNX Runtime gives the same per-pixel labels as PyTorch on a real frame.

  python scripts/export_onnx.py outputs/models/instrument_seg/best.pt outputs/models/tissue_seg/best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.segmentation.predictor import SegmentationPredictor  # noqa: E402


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:]) or ["outputs/models/instrument_seg/best.pt",
                                                              "outputs/models/tissue_seg/best.pt"]
    frame = cv2.imread(str(sorted(Path("video01").rglob("*_endo.png"))[0]))
    for ck in paths:
        p = SegmentationPredictor.from_checkpoint(ck, "cpu")
        x = torch.from_numpy(p._prepare(frame))[None]
        out = Path(ck).with_name("model.onnx")
        torch.onnx.export(p.model, x, str(out), input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False)
        with torch.inference_mode():
            ref = p.model(x).numpy()
        p.use_onnx(out)
        o = p.predict_logits([frame]).numpy()
        agree = float((o.argmax(1) == ref.argmax(1)).mean())
        print(f"{out}: max|diff|={np.abs(o - ref).max():.2e}, label agreement {100 * agree:.4f}%")
        if agree < 0.9999:
            out.unlink()
            print("  REMOVED: outputs differ too much")
    return 0


if __name__ == "__main__":
    sys.exit(main())
