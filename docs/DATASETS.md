# Adding videos and patient-level splits

**Current status: all data is ONE video (Cholec80 video01, one procedure).** Every reported score is
within-patient. Do not claim patient-level generalisation until at least 2 more videos are added and
the test split is a different video.

## Sources (do not redistribute; accept each licence yourself)

* **CholecSeg8k** (Hong et al., 2020, arXiv:2012.12453): pixel-level labels for frames from several
  Cholec80 videos in the same `videoXX/videoXX_NNNNN/frame_N_endo*.png` layout as `video01/`.
  Published on Kaggle; check the dataset page for the exact video list and licence (stated as
  CC BY-NC-SA 4.0 at the time of writing; verify).
* **Cholec80** (CAMMA, University of Strasbourg, request at https://camma.unistra.fr/datasets/):
  80 full videos with phase and tool-presence labels only, **no pixel masks**. It can be used for
  DLC tip labelling and risk-event annotation, but not for segmentation training without new masks.

Each Cholec80 video is a separate procedure. Patient identity is not released, so the video is the
closest available proxy for the patient-level unit.

## Layout

Put new videos next to the existing data without moving `video01/`, e.g. `data/cholecseg8k/video09/…`.
`data.root` accepts a list:

```yaml
data:
  root: [video01, data/cholecseg8k/video09, data/cholecseg8k/video12]
  group_by: video        # split unit = video (default `clip` keeps the current behaviour)
```

With `group_by: video`, a video listed in two splits is an error (leakage). With `group_by: clip`,
a warning says the split is within-patient.

## Commands

```powershell
.venv\Scripts\python scripts\make_splits.py --out outputs\splits_proposal.yaml   # proposes a video-level split; refuses with 1 video
.venv\Scripts\python train_segmentation.py --config configs\instrument_seg.yaml data.group_by=video
.venv\Scripts\python evaluate_segmentation.py --config configs\instrument_seg.yaml --checkpoint outputs\models\instrument_seg\best.pt --split test
.venv\Scripts\python scripts\class_support.py --config configs\tissue_seg.yaml --checkpoint outputs\models\tissue_seg\best.pt
```

Paste the proposed `data.splits` from `outputs/splits_proposal.yaml` into a config before training.
On CPU, budget about 5 min per epoch per 880 training frames.
