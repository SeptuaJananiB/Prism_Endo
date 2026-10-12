# EndoVis'15 robotic-instrument keypoints — integration report

**Separate experiment.** These are robotic (da Vinci) claspers on ex vivo tissue. Their joints are
**not** mapped to PRISM-Endo's laparoscopic `lhook_tip` / `grasper_tip` / `grasper_tip_2`, and nothing
here changes the laparoscopic pipeline. A DLC model trained here is not used by PRISM-Endo until
compatibility with laparoscopic instruments is shown.

## Sources and licences

| Item | Location | Licence / access |
|---|---|---|
| Joint annotations (Du et al., IEEE TMI 2018) | `Downloads\PRISM_Endo_Public_Keypoint_Download_Kit\...\EndoVisPoseAnnotation-master` | MIT (repository LICENSE) |
| EndoVis'15 *Tracking_Robotic_Training* videos | zip `Downloads\Tracking_Robotic_Training.zip` (untouched), extracted to `Downloads\PRISM_Endo_ExternalData_Tracking\` | EndoVis'15 challenge data terms (https://endovissub-instrument.grand-challenge.org/); cite the challenge |
| EndoVis'15 *Tracking_Robotic_Testing* videos | **not present** | same challenge site, registration/login required; not downloaded (needs your approval and account) |

## Files discovered (verified)

* `train_labels/train1..4_labels.json` and `test_labels/test1..6_labels.json`. Each is a list with one entry per video
  frame (`filename` …`/DatasetK/Raw/img_<NNNNNN>_raw.png`, 1-based); labelled entries carry
  `annotations = [{class, id, x, y}]`.
* Joint classes found: the README's five (LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint)
  **plus an undocumented `TrackedPoint`** on the shaft. Tool ids `tool1` = right clasper, `tool2` = left clasper.
  **No visibility field**: an absent joint is the only "not visible" signal.
* `DatasetK/*Pose.txt`: one line per frame `x y` plus axis vectors and an angle, i.e. the **original EndoVis'15
  tracking annotation**, a different label type. It is **not** used as joint ground truth.
* Videos: Dataset1–4 `Video.avi`, 720×576, 25 fps, 1107 / 1125 / 1124 / 1123 decodable frames (match the README).

## Validation (`python scripts/endovis_validate.py` → `outputs/endovis/validation_report.json`)

| | Dataset1 | Dataset2 | Dataset3 | Dataset4 | Total |
|---|---|---|---|---|---|
| labelled frames | 210 | 240 | 252 | 238 | 940 |
| matched to a decodable frame | 210 | 240 | 252 | 238 | **940 (100 %)** |

* Coordinates outside the image: 0. Unknown classes / tool ids: 0.
* **3 frames with a duplicated joint** (same joint twice under `tool1`, missing under `tool2`; probably a wrong tool
  id): Dataset1 frames 130 and 180, Dataset2 frame 410. Flagged and **excluded**, not repaired.
* 28 frames where one tool lacks some joints: kept, with the joints left unlabelled (NaN).
* Gallery `outputs/endovis/gallery/`: points sit on the instruments (jaw tips, wrist, shaft) with
  `video index = frame number − 1`. This alignment check is **visual only**.
* Test labels: the 6 test JSON files exist, but their videos are not present, so 0 test frames can be matched.

## DLC conversion and split (`python scripts/endovis_to_dlc.py`)

* Project `outputs/dlc_projects/endovis-robotic-du2018-2026-10-10`, 12 bodyparts named after the source
  (`tool1_LeftClasperPoint` … `tool2_TrackedPoint`), `CollectedData_du2018.h5/.csv` (DLC ≥ 2.3 format).
* **Split by whole sequence:** Dataset1–3 → project (699 frames; DLC's own 95/5 shuffle is used for monitoring
  only). **Dataset4 (238 frames) is never added** and is used only for the final evaluation.
  The official test set is unavailable, so Dataset4 is a held-out *training-set sequence* (same setup/day),
  not the official test.

## Training and evaluation

```powershell
.venv\Scripts\python scripts\dlc_workflow.py build --config outputs\dlc_projects\endovis-robotic-du2018-2026-10-10\config.yaml
.venv-dml\Scripts\python scripts\dlc_train_gpu.py --config outputs\dlc_projects\endovis-robotic-du2018-2026-10-10\config.yaml --epochs <N>
.venv-dml\Scripts\python scripts\endovis_evaluate.py --config outputs\dlc_projects\endovis-robotic-du2018-2026-10-10\config.yaml
```

Timing: one CPU epoch (ResNet-50, 699 frames, batch 8) took ~11 min while the GPU was also busy.
GPU (DirectML) epoch time: 274 s (measured).

## Results (measured 2026-10-10, held-out Dataset4, never in training)

Model: DLC 3.0.2, ResNet-50, trained **19 epochs on the AMD GPU via DirectML** (4,703 s; one epoch = 274 s, versus ~11 min
on CPU). Evaluated snapshot: `snapshot-best-010.pt` (DLC's own best on its internal split). Likelihood cutoff 0.6.
Errors are computed **only where the model made a confident prediction**, so read them together with recall.

| Joint | labelled | detection recall | mean px | median px | PCK@10px |
|---|---|---|---|---|---|
| tool1_LeftClasperPoint | 238 | 0.75 | 2.8 | 2.6 | 0.99 |
| tool1_RightClasperPoint | 238 | 0.71 | 12.6 | 3.2 | 0.88 |
| tool1_HeadPoint | 235 | 0.54 | 10.2 | 3.1 | 0.96 |
| tool1_ShaftPoint | 238 | 0.33 | 4.3 | 4.3 | 1.00 |
| tool1_EndPoint | 238 | 0.02 | 10.0 | 10.8 | 0.25 |
| tool1_TrackedPoint | 225 | 0.83 | 3.7 | 3.4 | 1.00 |

* Over all labelled tool1 joints: **745/1412 detected (53%)**. On detected points: mean **6.7 px**,
  median **3.1 px**, PCK@5 0.79, PCK@10 **0.96**, PCK@20 0.97.
* Dataset4 contains only `tool1` (right clasper), so the 6 `tool2` joints are not evaluable here.
* EndPoint (shaft end at the image border) is almost never confidently detected (recall 0.02).
* Inference: 1.75 fps on CPU (1,123 frames, 720×576).
* Caveats: 19 epochs is short (DLC's default is 200); Dataset4 is a held-out sequence of the *training* release, not the
  official test set; robotic instruments only, so these numbers do **not** describe laparoscopic L-hook/grasper tips.
* DirectML note: DLC snapshots written on DirectML must be converted with `scripts/dlc_snapshots_to_cpu.py` before
  DLC can load them (originals kept as `*.dml.bak`).

## Longer training (79 epochs, AMD GPU) — measured 2026-10-10

Resumed from epoch 19 → 40 (interrupted: background jobs were stopped when RAM ran low) → 79. DLC's best snapshot on its
internal split: **epoch 50** (`snapshot-best-050.pt`), evaluated on held-out Dataset4 (`outputs/endovis/eval_long/`).

| Joint (tool1) | recall 19 ep → 79 ep | median px | mean px | PCK@10 |
|---|---|---|---|---|
| tool1_LeftClasperPoint | 0.75 → **0.89** | 2.6 → 2.6 | 2.8 → 3.0 | 0.99 → 1.00 |
| tool1_RightClasperPoint | 0.71 → **1.00** | 3.2 → 4.4 | 12.6 → 117.6 | 0.88 → 0.63 |
| tool1_HeadPoint | 0.54 → **1.00** | 3.1 → 4.7 | 10.2 → 105.3 | 0.96 → 0.71 |
| tool1_ShaftPoint | 0.33 → **0.53** | 4.3 → 4.3 | 4.3 → 4.4 | 1.00 → 1.00 |
| tool1_EndPoint | 0.02 → **0.63** | 10.8 → 4.9 | 10.0 → 5.7 | 0.25 → 0.85 |
| tool1_TrackedPoint | 0.83 → **0.79** | 3.4 → 3.6 | 3.7 → 3.6 | 1.00 → 1.00 |

* Overall: detected **745/1412 (53%) → 1140/1412 (81%)**; median 3.1 → 3.8 px;
  PCK@10 0.96 → 0.84.
* Longer training mainly raised **recall**. The Right-clasper and Head joints now have occasional gross errors (mean ≫ median),
  consistent with confusing the two jaws or the two tools in some frames. That is the next thing to fix (more epochs with
  identity-aware augmentation, or a multi-animal DLC project with tool identities).

## Segmentation_Robotic_Training.zip (inspected, not extracted)

15 entries, 0.04 GB. Per dataset: `Video.avi` plus instrument mask videos (`Segmentation.avi`, or Left/Right for
Dataset1). The four `Video.avi` files are **byte-identical** (same CRC-32 and size) to those in
Tracking_Robotic_Training.zip, so this archive adds robotic instrument masks only, no new footage.

## Domain difference to PRISM-Endo

Robotic articulated claspers/scissors, ex vivo porcine tissue, 720×576, a different camera; versus laparoscopic
L-hook/grasper in human cholecystectomy (CholecSeg8k, 854×480). Joint definitions differ (e.g. a clasper has two jaw
tips and a wrist joint, while an L-hook has a single hook tip). Any transfer to PRISM-Endo needs laparoscopic
labels to test it.
