"""End-to-end check that the DeepLabCut integration works in this environment.

Uses the SYNTHETIC clip from make_synthetic_video.py, whose tip position is
known by construction, to write DLC-format labels programmatically. This
replaces the human labelling step ONLY for this code-path test; it says
nothing about accuracy on surgical video, and these labels must never be
mixed into a real project.

Steps exercised: create project -> extract frames -> write labels ->
create_training_dataset -> train (1-2 epochs) -> analyze_videos ->
read predictions with PRISM-Endo's DLC reader.

  python scripts/dlc_smoke_test.py --epochs 2
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_synthetic_video import main as make_video  # noqa: E402
from src.tracking import dlc_workflow as wf  # noqa: E402
from src.tracking.dlc_io import DLCPredictions  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="outputs/dlc_smoke")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    work = Path(args.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    video = work / "synthetic.mp4"
    make_video(["--out", str(video), "--frames", "60"])
    truth = pd.read_csv(work / "synthetic_tip_truth.csv").set_index("frame")

    cfg = wf.create_project("smoke", "synthetic", [video], work, ["lhook_tip"], numframes2pick=12)
    wf.extract_frames(cfg, algo="uniform")
    import deeplabcut
    from deeplabcut.utils.auxiliaryfunctions import read_config

    dcfg = read_config(str(cfg))
    scorer = dcfg["scorer"]
    lab_dir = Path(dcfg["project_path"]) / "labeled-data" / video.stem
    imgs = sorted(lab_dir.glob("img*.png"))
    if not imgs:
        raise RuntimeError(f"DLC extracted no frames into {lab_dir}")
    rows, index = [], []
    for img in imgs:
        f = int("".join(ch for ch in img.stem if ch.isdigit()))
        rows.append([truth.loc[f, "x"], truth.loc[f, "y"]])
        index.append(("labeled-data", video.stem, img.name))
    cols = pd.MultiIndex.from_product([[scorer], ["lhook_tip"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=cols)
    df.to_hdf(lab_dir / f"CollectedData_{scorer}.h5", key="keypoints", mode="w")
    df.to_csv(lab_dir / f"CollectedData_{scorer}.csv")
    print(f"wrote {len(rows)} synthetic labels")

    wf.build_dataset(cfg, "resnet_50")
    wf.train(cfg, epochs=args.epochs, batch_size=4, device=args.device, save_epochs=1)
    pred_file = wf.analyze(cfg, video, work / "analysis", device=args.device, labeled_video=False)
    pred = DLCPredictions(pred_file)
    x, y, lik = pred.get(30, "lhook_tip")
    print(f"DLC prediction file: {pred_file}")
    print(f"frame 30: predicted ({x:.1f}, {y:.1f}) lik={lik:.2f}; truth ({truth.loc[30, 'x']:.1f}, "
          f"{truth.loc[30, 'y']:.1f})  [after only {args.epochs} epochs - accuracy not meaningful]")
    print("DLC integration smoke test: OK (deeplabcut", deeplabcut.__version__, ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
