"""Open the next not-yet-labelled DLC frame folder in napari (napari-deeplabcut).

  .venv-label\\Scripts\\python scripts\\label_frames.py --project train      # next training folder
  .venv-label\\Scripts\\python scripts\\label_frames.py --project eval       # next held-out eval folder
  .venv-label\\Scripts\\python scripts\\label_frames.py --project train --video video01_16345

In napari: pick a bodypart in the keypoint panel, click the tip, move to the next frame; leave
invisible tips unlabelled; press Ctrl+S to save (writes CollectedData_annotator.h5 in the folder).
Rules: docs/DLC_ANNOTATION.md.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "outputs" / "dlc_projects"


def progress(project: Path) -> list[tuple[str, int, bool]]:
    rows = []
    for vid in sorted((project / "labeled-data").iterdir()):
        if vid.is_dir():
            n = len(list(vid.glob("img*.png")))
            rows.append((vid.name, n, any(vid.glob("CollectedData_*.h5"))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", choices=["train", "eval"], default="train")
    ap.add_argument("--video", help="specific labeled-data folder name (default: next one without labels)")
    args = ap.parse_args(argv)
    project = sorted(PROJECTS.glob(f"prism-tips-{args.project}-*"))[-1]
    rows = progress(project)
    print(f"{project.name}:")
    for name, n, saved in rows:
        print(f"  {'[saved]' if saved else '[todo] '} {name:16s} {n:3d} frames")
    todo = [r[0] for r in rows if not r[2]]
    target = args.video or (todo[0] if todo else None)
    if target is None:
        print("All folders have a saved CollectedData file. Check completeness with:\n"
              f"  .venv\\Scripts\\python scripts\\dlc_workflow.py status --config {project / 'config.yaml'}")
        return 0
    import napari

    viewer = napari.Viewer(title=f"PRISM-Endo labelling - {project.name} / {target}")
    viewer.open(str(project / "labeled-data" / target), plugin="napari-deeplabcut")
    viewer.open(str(project / "config.yaml"), plugin="napari-deeplabcut")
    print(f"Opened {target}. Label, then Ctrl+S. Close the window and rerun for the next folder.")
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
