# Instrument-tip annotation guide (DeepLabCut)

No tip labels exist yet. Everything below is done by a **human annotator**; nothing in this
repository generates tip labels. Machine suggestions are deliberately not provided, so that
heuristic points can never be saved as ground truth by accident.

## Projects (created by `scripts/prepare_dlc_annotation.py`)

| Project | Frames | Source clips | Use |
|---|---|---|---|
| `outputs/dlc_projects/prism-tips-train-annotator-<date>` | ~200 | TRAIN split only | label → train DeepLabCut |
| `outputs/dlc_projects/prism-tips-eval-annotator-<date>` | ~40 | TEST split only (14859–15099) | label → held-out evaluation **only, never train on it** |

Frames were chosen with the ground-truth *segmentation* masks only to make sure an L-hook and/or
grasper is visible, then for visual diversity. `selection_manifest.csv` lists every frame.

## Bodyparts

| Bodypart | Definition |
|---|---|
| `lhook_tip` | L-hook electrocautery: the **distal-most point of the metal hook** (the outer bend of the "L"). |
| `grasper_tip` | Grasper: the **midpoint between the two jaw tips**. If the jaws are closed, the single closed tip point. |
| `grasper_tip_2` | A second visible grasper, same definition. |

**Two-grasper identity rule:** `grasper_tip` = the grasper whose shaft enters the image from the
**left**; `grasper_tip_2` = the one entering from the **right**. If only one grasper is visible, use
`grasper_tip` for it whichever side it enters from. Apply this rule in every frame.

## When to leave a point unlabelled

* The tip is **occluded** (by tissue, the other instrument or smoke) or **outside** the field of view.
* The tip location is **ambiguous by more than ~10 px** (motion blur, glare).
* The instrument is not present.

Unlabelled = NaN = "not visible"; it is never treated as ground truth. Do **not** guess.

## Steps

```powershell
# GUI (not installed by default; ideally in its own environment)
.venv\Scripts\python -m pip install "deeplabcut[gui]==3.0.2"
.venv\Scripts\python scripts\dlc_workflow.py label  --config outputs\dlc_projects\prism-tips-train-annotator-<date>\config.yaml
.venv\Scripts\python scripts\dlc_workflow.py label  --config outputs\dlc_projects\prism-tips-eval-annotator-<date>\config.yaml
.venv\Scripts\python scripts\dlc_workflow.py status --config <project>\config.yaml     # counts human labels

# Training is refused until >=100 labelled frames and >=30 labels per bodypart exist
.venv\Scripts\python scripts\dlc_workflow.py build    --config <train project>\config.yaml
.venv\Scripts\python scripts\dlc_workflow.py train    --config <train project>\config.yaml --epochs 200 --device cpu
.venv\Scripts\python scripts\dlc_workflow.py evaluate --config <train project>\config.yaml

# Predict on the held-out test videos and score against the eval project's labels
.venv\Scripts\python scripts\dlc_workflow.py analyze --config <train project>\config.yaml --video Videos\video01_14859.mp4 --out outputs\dlc\video01_14859
.venv\Scripts\python scripts\evaluate_dlc_heldout.py --eval-config <eval project>\config.yaml --pred-dir outputs\dlc
# Baseline: the current mask-tip heuristic, from pipeline runs on the same videos
.venv\Scripts\python scripts\evaluate_dlc_heldout.py --eval-config <eval project>\config.yaml --pipeline-runs outputs\runs
```

On CPU, DLC training with a ResNet-50 backbone on ~200 frames takes hours for 200 epochs. Start
with a short run (e.g. `--epochs 20`) to check the loss falls. Quality control: have a second person
re-label ~20 eval frames, and report inter-annotator pixel error as the noise floor.

## Known data limitation: two graspers

Only **13 of 880 training-split frames** show two graspers, but **every** frame of the four test clips
does (measured from the ground-truth masks). `grasper_tip_2` therefore cannot be learned well from the
current training clips, so the label gate treats it as optional. Held-out evaluation scores grasper tips
identity-free (each prediction is matched to the nearest labelled grasper). Adding training clips that
show two graspers is the fix.

## Easier alternative: click tool (recommended)

```powershell
.venv\Scripts\streamlit run src\ui\label_tips.py --server.port 8601
```

One frame at a time with the instrument outlines drawn (pink = L-hook, blue = grasper, taken from the
ground-truth segmentation, which only helps you find the tools). Choose the tip, click it, then
**Save & next**. It writes the same `CollectedData_annotator.h5/.csv` files DeepLabCut reads, plus
`reviewed.json` so frames where no tip is visible still count as done.

## First human-labelled evaluation (2026-10-10)

The **40 held-out eval frames were labelled by a human** with `label_tips.py` (training project: still 0/200).
Label QA against the ground-truth segmentation (`outputs/label_qa/`, `flags.json`):
* 9 L-hook points lie 6–27 px beyond the L-hook mask. Visual check: they sit on the bare metal hook past the white
  sheath, which the segmentation does not include, so the label is plausible and kept.
* Excluded from scoring (labels **not** modified, listed in `outputs/label_qa/excluded.json` for review): clip 14939
  img04 and img16 (`lhook_tip` on the grasper; no L-hook in the mask) and img27 (`lhook_tip` and `grasper_tip` swapped).

Mask-tip method (instrument model that never saw these clips: `outputs/models/instrument_seg/best.pt`) versus
human labels (`scripts/evaluate_dlc_heldout.py --pipeline-runs outputs/runs_tipbaseline --exclude …`):

| Tip | labelled | detected | median px | mean px | PCK@10 | PCK@20 | PCK@5 %diag |
|---|---|---|---|---|---|---|---|
| L-hook | 37 | 37 (100 %) | **22.4** | 22.9 | 0.14 | 0.49 | 0.97 |
| Grasper (identity-free) | 49 | 48 (98 %) | 129.5 | 132.1 | 0.00 | 0.00 | 0.00 |

* L-hook: a consistent ~20 px shortfall, because the mask (and so the method) stops at the insulated sheath.
* Grasper: **not a valid accuracy number yet**. The human points are near the jaw hinge/base, while the guide and the
  method use the far jaw ends (`outputs/label_qa/compare_grasper.jpg`). The grasper tip definition must be agreed and
  the grasper points relabelled consistently before this can be reported.
* The 200 training frames are still unlabelled, so no laparoscopic DLC model is trained. A pseudo-label project
  (`prism-tips-PSEUDO-*`, `scripts/make_pseudo_labels.py`, machine-made) exists but was deliberately **not** used for
  training, because it would reproduce the method's own biases.

### Prototype DLC trained on machine (pseudo) labels — result: rejected (2026-10-10)

Trained 30 epochs (ResNet-50; epochs 1–15 CPU, 16–30 AMD GPU) on the 198 machine-labelled frames of
`prism-tips-PSEUDO-*`, then predicted the 4 held-out test clips and scored against the **40 human labels**
(`outputs/logs/pseudo_dlc_eval.log`):

| Tip | detected | median px | PCK@20 |
|---|---|---|---|
| L-hook | 14/37 (38 %) | 339 | 0.36 |
| Grasper | 8/49 (16 %) | 438 | 0.00 |

This is far worse than the mask-tip method it was trained to imitate (L-hook 100 % detected, median 22 px), probably
because of too few frames (198) and a scene shift between the training and test clips. **Not used by the pipeline.**
A useful laparoscopic DLC model needs the 200 training frames labelled by a human (plus more videos).

## Automatic labelling pipeline and new default tip method (2026-10-10, evening)

**Grasper point definition: jaw HINGE** (where the jaws meet the shaft). The 40 human eval labels use it, and it is
less ambiguous for open jaws than "jaw ends". The old 129 px grasper error was this definition mismatch.

`scripts/auto_label_tips.py` (method `auto_geo_v1`, `src/tracking/auto_tips.py`) labels the 200 training frames from
the GROUND-TRUTH CholecSeg8k instrument masks, with no clicking:
* L-hook tip = geodesic far end of the L-hook mask from its field-of-view entry;
* grasper hinge = 90 px back from the geodesic far end along the instrument (`jaw_px` tuned 20–140 px on the TRAINING
  frames; the human eval frames were not used for tuning);
* absent instrument → missing (no coordinates); each point stores confidence, method, distance to the previous rule and
  to the old rule, temporal jump to the neighbouring frame, and review flags (`outputs/auto_labels/`:
  `train_auto_labels.csv`, `summary.json`, `contact_sheet_1-5.jpg`). **These are pseudo-labels, not human-verified.**

Labeller vs the 40 human frames (GT masks, measured once after freezing): L-hook median 11.7 px, grasper hinge
median 17.7 px (15.5 px with the half-resolution search; PCK@20 0.63–0.69; some fragment failures, p90 387 px).

### Pipeline result (predicted masks, leakage-free single-video instrument model, min-conf 0)

| Keypoint method | L-hook found / median | Grasper found / median / p90 | keypoint ms/frame |
|---|---|---|---|
| Old rule (straight-line far end) | 37/37 · 22.4 px | 49/49 · 128.3 px · 143.8 px | 15 |
| Geodesic far end for both | 37/37 · 23.4 px | 49/49 · 32.1 px · 49.9 px | ~220 (full-res BFS) |
| **New default: grasper hinge (half-res geodesic), L-hook old rule** | **37/37 · 22.4 px** | **49/49 · 31.8 px · 48.9 px** | **~45** |

The multi-video instrument model was trained on all of video01, which contains these 4 clips, so it was NOT used for
this comparison. Config: `keypoints.mask_tip_landmarks` in `configs/pipeline.yaml`.

### DLC trained on the automatic labels — not adopted

`scripts/run_auto_label_dlc.sh`: 60 epochs on the AMD GPU (3257 s), final-epoch snapshot, 4 held-out clips, human
labels (`outputs/logs/auto_dlc_eval_c{0.5,0.1}.log`). At conf ≥ 0.5: L-hook 0/37 detected, grasper 23/49 (median
40.7 px, p90 508 px). At conf ≥ 0.1: L-hook 31/37 but median 206 px. **The mask-based method stays the default**, and
DLC is not better on this data (200 frames from one video).


## Overnight autopilot result (2026-10-11 07:15:54)

DLC v2 (1,387 auto labels, 16 videos, epochs 31-80) + settling phase (low learning rate) and hybrid variants, scored on the 40 human frames (median px, no confidence cut-off):

| Option | L-hook | Grasper |
|---|---|---|
| dlc_only | 74.05720275026137 | 97.09691957862722 |
| hybrid_all | 22.44484796116915 | 40.63546178132327 |
| hybrid_grasper | 22.44484796116915 | 40.63546178132327 |
| hybrid_lhook | 22.44484796116915 | 31.889433874962975 |
| geometry_current | 22.4 | 31.9 |

Rule: adopt a hybrid only if neither tip is >1 px worse, both recalls 100 %, and the sum of medians improves. **Choice: geometry_current.** App default unchanged.
Caveat: all options compared on the same 40 human frames (video01) used throughout; small set.

## DLC v3 with HUMAN L-hook clicks → hybrid adopted (2026-10-11)

Your 120-frame L-hook session (6 videos, video01 excluded): 96 saved (86 tips, 10 not visible). Human vs automatic
L-hook labels: median 11.8 px, no systematic offset, but in some frames the CholecSeg8k mask omits the thin metal hook
so the automatic label stops 20–50 px short (`outputs/label_qa/lhook_offmask_cases.jpg`).
DLC v3 (`scripts/make_dlc_v3_project.py`): L-hook only from the 86 human clicks (+10 negatives), frames with automatic
L-hook labels dropped, automatic grasper labels kept; 25 epochs on the AMD GPU from the settled v2 model
(lr 1e-4, last 5 epochs 2e-5). Final model stored in `outputs/models/dlc_tips_v3/` (epoch 115).

| Option (40 human frames, median) | L-hook | Grasper |
|---|---|---|
| DLC v3 alone (final) | 69.6 px | 161.9 px |
| DLC v3 epoch 110 (not used: would be test-set selection) | 31.8 px | 133.1 px |
| Hybrid, DLC for both tips | 18.4 px | 32.3 px |
| **Hybrid, DLC for L-hook only (NEW DEFAULT)** | **18.4 px** (PCK@20 54 %) | **31.9 px** |
| Previous default (geometry) | 22.4 px | 31.9 px |

`configs/pipeline.yaml`: `keypoints.source: hybrid`, `hybrid.dlc_slots: [lhook_tip]`, `auto_analyze: true` — for a new
video the pipeline runs the stored DLC model once (CPU, cached in `outputs/dlc/auto_analyze/`), checked end to end
(identical points to the evaluated run). Previous config: `configs/pipeline.before_autopilot.yaml`.
Caveat: chosen with a rule fixed in advance, but compared on the same 40 frames used throughout (small set).

## Wider test on other patients (video52 + video55 held out), 2026-10-11

DLC trained exactly like v3 but WITHOUT any video52/55 frames or clicks (`prism-tips-V3fold-holdout5255-*`), scored on
your 35 L-hook clicks from those two videos (`scripts/eval_lhook_fold.py`, `outputs/fold_eval/fold_results.json`).
Both methods use the single-video instrument model (never saw these videos).

| Method | Found | Median | Mean | Within 20 px |
|---|---|---|---|---|
| Geometry (mask tip) | 25/35 | 28.6 px | 60.5 px | 44 % |
| **DLC / hybrid (L-hook)** | **35/35** | **9.8 px** | **12.3 px** | **80 %** |

Caveat (honest): the fold model was initialised from the settled v2 model, which had been trained on AUTOMATIC labels of
all 16 videos including video52/55 — so the network had seen these patients' images (with the shorter automatic
L-hook points, not your clicks). The 9.8 px is therefore optimistic; a fully clean check needs an initialisation that
never saw video52/55.

## Wider grasper test (human jaw-hinge clicks, 5 videos 17/25/28/52/55, 40 frames, 44 points) — 2026-10-11

`scripts/eval_grasper_wider.py` → `outputs/grasper_eval/grasper_wider_results.json` (single-video instrument model).

| Method | Found | Median | PCK@40 |
|---|---|---|---|
| Geometry, jaw hinge (default) | 23/44 | 126 px | 22 % |
| Geometry, jaw end (old) | 33/44 | 155 px | 9 % |
| Labeller on ground-truth masks | 38/44 | 75 px | 29 % |

Finding: the grasper method does **not** generalise. The hinge rule uses a fixed 90 px jaw length tuned on video01, but
the apparent jaw size changes strongly with zoom between videos (`outputs/grasper_eval/check.jpg`), and the
single-video instrument model misses many graspers (21/44 not found). The video01 result (31.9 px) therefore does not
represent other patients. Fix direction (not done): scale-aware jaw length (e.g. relative to shaft width) and a
grasper keypoint model trained on human hinge clicks from several videos.

**Correction (same scoring rule as video01: confident + low-confidence points count):** geometry hinge 40/44 found,
median 66.9 px, PCK@40 30 %; scale-aware hinge 124.3 px; old jaw-end 113.5 px; labeller on ground-truth masks 75.1 px.
Diagnosis: misses were keypoint confidence gating, not segmentation (only 4/38 had no predicted mask). The default
(fixed hinge) remains the best of the geometric variants; the scale-aware variant is not adopted. Even perfect masks give
75 px, so the jaw hinge cannot be derived reliably from mask geometry across zoom levels: a learned grasper keypoint
model trained on human hinge clicks from several videos is required (no such training labels exist yet — the 49 video01
and 44 multi-video points are test sets).
