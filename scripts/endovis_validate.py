"""Validate EndoVis'15 joint annotations against the source videos and draw a sample gallery.

  python scripts/endovis_validate.py
Writes outputs/endovis/validation_report.json and outputs/endovis/gallery/*.jpg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.external.endovis import draw, load_labels, read_frame, validate  # noqa: E402

DL = Path.home() / "Downloads"
ANN = DL / "PRISM_Endo_Public_Keypoint_Download_Kit/PRISM_Endo_ExternalData/EndoVisPoseAnnotation/EndoVisPoseAnnotation-master"
TRAIN = DL / "PRISM_Endo_ExternalData_Tracking/Tracking_Robotic_Training/Training"
TEST = DL / "PRISM_Endo_ExternalData_Tracking/Tracking_Robotic_Testing"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default=str(ANN))
    ap.add_argument("--train-dir", default=str(TRAIN))
    ap.add_argument("--test-dir", default=str(TEST))
    ap.add_argument("--out", default="outputs/endovis")
    ap.add_argument("--gallery", type=int, default=4, help="sample frames per sequence")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "gallery").mkdir(parents=True, exist_ok=True)
    report = {}
    for split, d in (("train", Path(a.train_dir)), ("test", Path(a.test_dir))):
        frames, file_stats = [], {}
        for js in sorted((Path(a.annotations) / f"{split}_labels").glob("*.json")):
            fr, st = load_labels(js, split)
            frames += fr
            file_stats[js.name] = st
        videos = {int(p.parent.name.replace("Dataset", "")): p for p in d.glob("Dataset*/Video.avi")} if d.is_dir() else {}
        v = validate(frames, videos)
        v["files"] = file_stats
        v["video_dir_present"] = d.is_dir()
        report[split] = v
        for seq in sorted(videos):
            lab = [f for f in frames if f.seq == seq and f.points]
            for f in lab[:: max(1, len(lab) // a.gallery)][: a.gallery]:
                img = read_frame(videos[seq], f.frame_number)
                if img is not None:
                    cv2.imwrite(str(out / "gallery" / f"{split}_seq{seq}_frame{f.frame_number:04d}.jpg"), draw(img, f))
        print(f"[{split}] video folder present: {d.is_dir()} | summary: {v['summary']}")
        for k, s in v["per_sequence"].items():
            print(f"   Dataset{k}: {s}")
        for k, info in v["videos"].items():
            print(f"   video Dataset{k}: {info}")
    (out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"flagged frames: train {len(report['train']['flagged'])}, test {len(report['test']['flagged'])}")
    print(f"wrote {out / 'validation_report.json'} and gallery")
    return 0


if __name__ == "__main__":
    sys.exit(main())
