# PRISM-Endo — Project status report (audited 2026-10-10, 04:46–05:10)

Everything below was checked against files, checkpoints, logs and test runs during this audit.
"Verified" means re-run or re-inspected now; a README statement alone is not counted as proof.

## 1. Executive summary

* The **video pipeline works end-to-end on CPU**: segmentation → tip estimate → motion → proximity → rule-based risk →
  video/CSV/JSON. Measured now on an **unseen patient** (video17): **3.37 fps** default, **7.67 fps** with
  `configs/pipeline_cpu_fast.yaml` + ONNX Runtime. Not real-time.
* **Segmentation was retrained on 17 CholecSeg8k videos** with a video-level split. On **3 unseen patients** it clearly
  beats the old single-video models (tissue mean Dice **0.735 vs 0.406**, grasper **0.852 vs 0.744**).
  These are now the pipeline defaults.
* **Laparoscopic tips**: the 40 held-out frames are now human-labelled (training 0/200). Mask-tip method vs human:
  **L-hook median 22.4 px** (100 % detected; it stops ~20 px short at the insulated sheath). The grasper number (129 px)
  is **not valid**: the human labels use the jaw hinge while the method uses the jaw ends, so the definition must be
  agreed first. A prototype DLC trained on machine labels was tested and **rejected** (L-hook 38 % detected,
  median 339 px vs human labels); the mask-tip method stays the default.
* **Learned risk is not trainable**: there are no expert event labels. The rule-based score is an uncalibrated heuristic.
* **EndoVis robotic keypoints (separate experiment)**: 940/940 human-labelled frames matched and converted; DLC trained
  79 epochs on the AMD GPU (best epoch 50); on held-out Dataset4: **81% of labelled joints detected (19 ep: 53%); median
  3.8 px, PCK@10 0.84 on detected points; occasional gross jaw/tool confusions**. Robotic instruments, not used by PRISM-Endo.
* Tests: **87 passed, 0 failed** (baseline before changes: 86 passed).

## 2. Completed work (with evidence)

| Component | Evidence |
|---|---|
| Instrument segmentation, 17 videos | `outputs/models/instrument_seg_multivideo/best.pt` (epoch 12, sha256 `8dea55259c08…`); `eval_test/metrics.json`: grasper Dice **0.852** / IoU 0.742 on videos 17, 25, 28 (1,200 frames) |
| Tissue segmentation, 17 videos | `outputs/models/tissue_seg_multivideo/best.pt` (best epoch 8 of 11 run; sha256 `862223dd739e…`); test Dice: fat 0.933, abdominal wall 0.874, instrument 0.874, liver 0.842, gallbladder 0.842, GI tract 0.780; mean **0.735** |
| Old single-video models (kept) | `instrument_seg/best.pt` (`226f7c2dc79a…`), `tissue_seg/best.pt` (`dcea016ab45e…`); on the same unseen videos: grasper 0.744, tissue mean 0.406 (`eval_unseen_videos/metrics.json`) |
| Leakage control | `scripts/compare_datasets.py`: local `video01/` is byte-identical to the downloaded one (5,120/5,120 files) and is used once; video-level split `outputs/splits/video_splits.yaml` (train 11 / val 3 / test 3 videos, no overlap) |
| Pipeline | runs on unseen video17 (`outputs/profile/unseen17_*`): grasper tip `ok` in 68–69/80 frames, honest `missing`/`low_confidence` otherwise, risk `Unavailable` when no valid tip |
| Target reliability | `class_support.json`: liver evaluated; cystic duct refused as a default target (see §8) |
| ONNX CPU backend | `model.onnx` for all four models; 100 % label agreement with PyTorch; ~2× faster forward |
| Temporal smoothing / camera motion | measured earlier on video01 test clips (`outputs/eval_temporal/temporal_test.json`); the code is covered by tests |
| UI tools | `src/ui/app.py`, `training_monitor.py`, `label_tips.py`, `annotate_events.py`: checked headlessly with streamlit.testing |
| Tests | `pytest tests`: 87 passed |

## 3. Partially completed

* **Tip keypoints:** the mask heuristic works (and handles two graspers), but its accuracy is unmeasured (no human tip labels).
* **L-hook segmentation on unseen patients:** test videos lack it; on the unseen validation videos Dice is 0.840 (see §11).
* **Tissue model 17-video run:** stopped at epoch 11/12 by a laptop shutdown; best checkpoint (epoch 8) used.
* **CPU fast preset:** speed measured; its effect on risk output is **not** equivalent (on video17 it gave 13 "High"
  frames vs 0 with default settings). Its accuracy has not been validated.
* **EndoVis integration:** validated and converted; a DLC training set is built; not trained or evaluated.

## 4. Pending (no blocker other than time/compute)

1. Optional: train EndoVis DLC longer (19 epochs so far; recall is limited) with `scripts/dlc_train_gpu.py --epochs 100`.
2. Re-split or add a test video that contains the L-hook, to evaluate it on unseen patients.
3. Measure the accuracy effect of `tissue.every_n_frames=2` and `nearest` upsampling on unseen videos.
4. Optionally finish tissue epoch 12 (`--resume …/tissue_seg_multivideo/last.pt`).

## 5. Blocked

| Task | Blocker |
|---|---|
| Laparoscopic DLC training | 200 training frames unlabelled (eval 40 done); the grasper tip definition must be agreed |
| Learned risk model, warning metrics | no surgeon event labels (schema, validator and annotation UI ready) |
| Cystic-duct reliability | all 239 training frames come from one patient; only 2 held-out frames exist (Dice 0.000 there) |
| Hepatic vein / liver ligament | training frames exist now (316 / 240) but no held-out frames, so unevaluated |
| EndoVis official test set | test videos are not downloaded (challenge registration needed); the 6 test JSONs cannot be matched |
| PhaKIR or other restricted data | not requested or downloaded |
| Long GPU runs | the laptop shut down twice under sustained GPU load (power/thermal); keep it plugged in and cooled |

## 6. Dataset status

| Dataset | On disk | Labels | Used |
|---|---|---|---|
| CholecSeg8k (17 videos, 8,080 frames) | `data/cholecseg8k/` (+ zip) | dense masks, 13 classes | yes, video-level split |
| Local `video01/` | duplicate of CholecSeg8k video01 | same | used once |
| EndoVisPoseAnnotation (MIT) | `Downloads\PRISM_Endo_Public_Keypoint_Download_Kit\…` | human joints: 940 train frames; 910 test frames (no videos) | converted, separate experiment |
| EndoVis'15 Tracking_Robotic_Training | zip + extracted `Downloads\PRISM_Endo_ExternalData_Tracking\` | 4 videos (720×576, 25 fps), `Pose.txt` = original tracking labels | matched 940/940 |
| EndoVis'15 Segmentation_Robotic_Training | zip only (not extracted) | instrument mask videos; its `Video.avi` files are **byte-identical** (CRC) to the tracking zip's | not needed yet |
| PRISM tip-annotation projects | `outputs/dlc_projects/prism-tips-{train,eval}-*` | 0/200 and 0/40 labelled | blocked |

## 7. Model status

| Model | State |
|---|---|
| instrument_seg_multivideo | trained, evaluated on unseen videos, **default** |
| tissue_seg_multivideo | trained (11/12 epochs), evaluated on unseen videos, **default** |
| instrument_seg / tissue_seg (video01 only) | trained, kept for comparison |
| Legacy notebook `models/unet_model.pth` | wrong target ("any annotated pixel"); not used |
| DLC laparoscopic tips | **untrained** (no labels) |
| DLC EndoVis robotic | trained 79 epochs (DirectML, best ep 50); held-out Dataset4: recall 81%, median 3.8 px (separate experiment) |
| Learned risk | **untrained** (no labels) |

## 8. Evaluation status

Measured: segmentation Dice/IoU on 3 unseen patients; within-patient numbers (README §10); flicker; CPU fps.
Not measurable yet: tip pixel error / PCK (no labels); L-hook on unseen patients; cystic duct (2 frames);
connective tissue, blood, hepatic vein and liver ligament on unseen patients (absent from test videos); risk
precision/recall/AUROC/lead time (no labels); EndoVis keypoints on the official test set (videos not downloaded); tool2 joints on Dataset4 (not present).

## 9. Prioritised next steps

1. **Get the 240 tip frames labelled** (any careful annotator, ~2 h with `label_tips.py`). This unlocks real
   laparoscopic keypoint training and evaluation.
2. Optional: longer EndoVis DLC training (GPU) to raise detection recall; re-evaluate on Dataset4.
3. Add an L-hook-containing unseen test video (re-run `scripts/make_splits.py` with another seed or add videos).
4. Surgeon defines and labels risk events (`src/ui/annotate_events.py`).

## 10. Commands to continue

```powershell
# 1. label tips (human)
.venv\Scripts\streamlit run src\ui\label_tips.py --server.port 8601
.venv\Scripts\python scripts\dlc_workflow.py status --config outputs\dlc_projects\prism-tips-train-annotator-2026-10-10\config.yaml
# 2. EndoVis DLC on the AMD GPU (274 s/epoch), convert snapshots, evaluate on held-out Dataset4
.venv-dml\Scripts\python scripts\dlc_train_gpu.py --config outputs\dlc_projects\endovis-robotic-du2018-2026-10-10\config.yaml --epochs 100
.venv-dml\Scripts\python scripts\dlc_snapshots_to_cpu.py outputs\dlc_projects\endovis-robotic-du2018-2026-10-10\dlc-models-pytorch
.venv\Scripts\python scripts\endovis_evaluate.py --config outputs\dlc_projects\endovis-robotic-du2018-2026-10-10\config.yaml
# pipeline on an unseen patient, fast CPU preset
.venv\Scripts\python run_pipeline.py --input data\cholecseg8k\video17\video17_01563 --config configs\pipeline_cpu_fast.yaml
# tests
.venv\Scripts\python -m pytest tests -q
```

## 11. Final round (2026-10-10, 13:30–14:00): measured

| Item | Result |
|---|---|
| **L-hook on unseen patients** (val videos 20/26/43, never trained on; used only to pick the best epoch) | **Dice 0.840** (17-video model) vs 0.544 (old single-video model); grasper 0.918 vs 0.741 |
| Tissue model | all 12/12 epochs done; epoch 12 (val 0.888) did not beat epoch 8 (0.907), so results are unchanged (test mean Dice 0.735) |
| Cystic duct | reliability `insufficient_test_data` (2 held-out frames); removed from default targets |
| Fast CPU preset vs default (3 unseen clips) | 2.3× faster (~8.1 vs 3.6 fps); same risk category in 80 / 100 / 100 % of frames; ≥2-level disagreement 1.2 % on one clip; proximity distance median difference < 1 px. Acceptable for a prototype |
| EndoVis DLC, 79 epochs | held-out recall 53 % → **81 %**, median 3.8 px; occasional jaw/tool confusions |
| Laparoscopic tips | mask-tip method vs 40 human labels: L-hook median 22.4 px. DLC trained on machine labels: rejected (338 px). |

**Not done, by design:** the learned risk model (it needs real risk-event labels; machine labels would only re-learn the
rule-based score) and validated grasper tip accuracy (the definition is unresolved: jaw ends vs jaw hinge).

## 12. DeepLabCut focus round + overnight autopilot (2026-10-10 evening → 2026-10-11 07:22)

All numbers: median error on the 40 human-labelled frames (video01), no confidence cut-off.

| Model | L-hook | Grasper | Note |
|---|---|---|---|
| **Current default (geometry, grasper hinge)** | **22.4 px** | **31.9 px** | kept |
| DLC #4 (1,190 auto labels, 16 videos, 30 ep) | 45.8 px | 64.8 px | |
| DLC v2 ep45 / ep50 (L-hook-balanced, 1,387 frames) | 34.7 / 249 px | 18.9 / 17.5 px | best snapshots, backed up; unstable |
| DLC v2 ep80 | 155.1 px | 76.1 px | |
| DLC settled ep90 (lr 5e-5) | 74.1 px | 97.1 px | |
| Hybrid (DLC on instrument, else geometry) | 22.4 px | 40.6 px | DLC L-hook never accepted |

Decision (rule fixed before results): no DLC option beat the default on both tips → **app default unchanged**.
Findings: DLC trained on automatic labels learns the training videos (internal RMSE 114 → 24 px) but does not
generalise reliably to a new patient video; results swing between checkpoints at lr 5e-4; DLC resume silently
restores the saved optimizer lr (fixed by stripping optimizer state). The ep45/ep50 snapshots were good, but
picking them would mean choosing on the test frames, so they are not adopted.
Most promising next step: human L-hook (and grasper) clicks on frames from videos other than video01.
Infrastructure: Windows scheduled supervisor (every 5 min) restarts services / resumes DLC / runs scoring; status
board in `status/`; hybrid keypoint source (`keypoints.source: hybrid`) available; tests 108 passing.

## 13. Learned safety model: Critical View of Safety (2026-10-11)

No downloadable surgical risk-event data exists (IEEE iAE dataset: no files uploaded; SAGES CVS: not public;
CholecTrack20: needs the user's data-use agreement). The learned safety model therefore uses Endoscapes-CVS201
(surgeon CVS labels, official video split 120/41/40). ResNet-50, trained on a free Google Colab T4 (RAM-cached frames +
fp16: 192 → 60 s/epoch), model selected on validation (epoch 8), test scored once:
**test mAP 0.510** (C1 0.470, C2 0.438, C3 0.621; balanced accuracy 0.68–0.72) — in the range of published
single-frame ResNet-50 baselines, below state of the art (~0.6). It answers "is CVS achieved?" (safe to clip), not
tissue-injury risk; it complements, not replaces, the rule-based proximity risk score. Code: `scripts/train_cvs.py`
(local DirectML) / `outputs/cloud/cvs_fast.py` + notebook (Colab). Report: `outputs/models/cvs_resnet50/cvs_report.json`.

**Update (CVS v2):** 320×576 input, dropout 0.3, rotation augmentation, 20 epochs on the Colab T4. Chosen over v1 by
VALIDATION mAP (0.528 vs 0.484); **test mAP 0.539** (C1 0.521, C2 0.514, C3 0.581) on the 40 unseen test videos.

## 14. Learned bleeding detector (2026-10-11)

SurgBlood public test release (20 laparoscopic clips, 886 frames, surgeon-reviewed bleeding masks; academic use),
split by clip in advance (13 train / 3 val / 4 test). U-Net ResNet-18 trained on a Colab T4, selected on validation
(epoch 12), test scored once (230 frames, 4 clips): **frame-level detection precision 0.974, recall 0.902, F1 0.937,
AUROC 0.942** (5 false alarms on 25 non-bleeding frames); bleeding-region Dice 0.671, IoU 0.574.
It detects visible bleeding (an adverse event); it does not predict injury, and transfer to CholecSeg8k videos is not
validated. Weights: `outputs/models/bleeding_unet/` (scripts/train_bleeding.py; Colab copy outputs/cloud/bleed_colab.py).

## 15. Tissue model for other surgeries, clean DLC test, web app (2026-10-11 → 12)

- **Clean DLC evaluation:** leakage-free L-hook test, 6.5 px median, 35/35 found (docs/DLC_CLEAN_EVALUATION.md).
- **Tissue + DSAD:** two fine-tunes (Colab T4 and a guarded local run). Both improve other surgeries
  (DSAD test 0.38 → 0.84) but lose ~0.04 Dice on gallbladder surgery, so the original stays default and the DSAD model
  is selectable as "Surgery type: Other" (configs/pipeline.yaml `tissue.by_surgery`).
- **Bleeding detector** wired into the pipeline, overlay and UI.
- **Web app (:8510, main interface):** React + FastAPI with synced video/timeline, events, reviewer flags, compare,
  stop, tissue-annotation options; model caching (settings change starts in 0.3 s), stop in ~0.2 s.
  Verified by 20 Playwright tests (all controls, light/dark, 4 screen sizes, WCAG 2 AA axe scan).
- Not done: clinical sign-off, learned grasper model (needs ~100 human hinge clicks), true mm (needs calibration),
  real-time speed (~5–6 fps on the RX 6800M), permanent public URL (needs an account).
