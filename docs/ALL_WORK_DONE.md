# PRISM-Endo — complete inventory of work done (as of 2026-10-10, ~21:00)

Everything below was verified against files in the project. Numbers are measured on held-out data unless marked.
"Before" = what existed when we started; everything else was built in our sessions.

## 0. What existed before

| Item | State found |
|---|---|
| `01_Environment_Setup.ipynb` | environment setup notebook |
| `02_Instrument_Segmentation(Fail).ipynb` | failed first attempt (marked by its author) |
| `02_Instrument_Segmentation.ipynb` + `models/unet_model.pth` | U-Net on 1 video; target was "any annotated pixel", not instruments → instrument Dice **0.26 (128 px) / 0.32 (256 px)** (`outputs/legacy_unet_eval/`) |
| `Dataset_Utilities/` | loaders + synthetic dataset; 2 of 5 tests failed (sample files never shipped) |
| `files_dataset/` | reference scaffold, no models/evaluation |

## 1. Segmentation

1. Segmentation framework: `src/segmentation/{models,train,predictor,metrics,optim,temporal}.py`, `train_segmentation.py`,
   `evaluate_segmentation.py`, `predict_segmentation.py`, configs `seg_base.yaml`, `instrument_seg*.yaml`, `tissue_seg*.yaml`.
2. Single-video instrument + tissue models (`outputs/models/instrument_seg`, `tissue_seg`).
3. Multi-video data loading (list of roots, `group_by: video`) — `src/data/segmentation_data.py`.
4. Video-level split of 17 videos — `scripts/make_splits.py` → `outputs/splits/video_splits.yaml` (train 11 / val 3 / test 3).
5. Retrained on 17 videos on the AMD GPU (DirectML): `instrument_seg_multivideo`, `tissue_seg_multivideo`.
   Unseen-patient test: grasper Dice **0.852**, L-hook **0.840** (val), tissue mean **0.735** (old 0.406).
6. Tissue epoch 8 → 12 continuation (no gain, kept epoch 8).
7. DirectML fixes: own soft-Dice loss, `SimpleAdamW` (avoids CPU fallback), `num_workers=0`, gradient accumulation,
   resume with `--max-restarts`.
8. Class reliability (`scripts/class_support.py`, `src/tissue/support.py`): evaluated / insufficient / unevaluated /
   untrained / failed per class; cystic duct, hepatic vein, liver ligament flagged.
9. Temporal smoothing (EMA α 0.7) — flicker −21 % tissue, −12 % instrument (`scripts/evaluate_temporal.py`).
10. Legacy model check — `scripts/evaluate_legacy_unet.py` (proved the old model's target was wrong).
11. ONNX export + ONNX Runtime CPU backend — `scripts/export_onnx.py`, `model.onnx` for all 4 models.

## 2. Instrument tips / keypoints

12. Mask-tip method (FOV entry → far end), multi-instance Hungarian slot tracking for 2 graspers — `src/tracking/mask_tip.py`, `sources.py`.
13. DeepLabCut integration: I/O, workflow CLI, label status gates — `src/tracking/dlc_io.py`, `dlc_workflow.py`, `scripts/dlc_workflow.py`, smoke test.
14. Annotation preparation: 200 training + 40 eval frames, diverse selection — `scripts/prepare_dlc_annotation.py`, `src/tracking/annotation_prep.py`.
15. Labelling tools: napari env (`.venv-label`), `scripts/label_frames.py`, browser click tool `src/ui/label_tips.py`.
16. Your 40 human eval labels; QA flagged 4 points (`outputs/label_qa/excluded.json`).
17. Held-out tip evaluation — `scripts/evaluate_dlc_heldout.py` (identity-free grasper matching, exclusions).
18. Experiment: DLC on machine pseudo-labels → rejected (L-hook 339 px).
19. Experiment: AI-click pilot (10 frames) → L-hook 15 px, not adopted.
21. Grasper definition resolved: **jaw hinge** (your convention).
22. Automatic labeller `auto_geo_v1` — `src/tracking/auto_tips.py`, `scripts/auto_label_tips.py` (200 frames, confidence,
    flags, contact sheets); vs your labels: L-hook 11.7 px, grasper 15.5–17.7 px.
23. Experiment: DLC on automatic labels (60 epochs GPU) → rejected (grasper 40.7 px, L-hook not detected).
24. **New default grasper method** (geodesic hinge, half-res, fallback): grasper **128 → 32 px**, L-hook 22 px.
25. Regression found and fixed (risk "Unavailable" 57/80 → 10/80 frames on video17).
26. Leakage check: multi-video model saw video01 → tip comparisons use the single-video model.
27. Auto-labels for **6,800 frames / 16 videos** (`scripts/auto_label_all.py`; video01 excluded).
28. TipNet heatmap network (backprop, GPU) — `src/tracking/tipnet.py`, `scripts/train_tipnet.py`,
    `scripts/eval_tipnet_human.py`; target check, bias init, resume/throttle options. **Not converged yet (in progress).**

## 3. Motion, proximity, risk

29. Kinematics (speed, direction, gap reset, EMA/Kalman) — `src/motion/kinematics.py`.
30. Camera motion (LK + RANSAC), compensated speed columns — `src/motion/camera.py`.
31. Proximity in pixels, FOV-border exclusion, per-frame cache, reliability flags — `src/proximity/proximity.py`.
32. Rule-based risk score with dwell time, rapid-rise escalation, explanations — `src/risk/rule_based.py`.
33. Learned risk model code with label gate — `src/risk/learned.py`, `scripts/train_risk_model.py` (not trained: no expert labels).
34. Risk-event schema, validator, inter-rater agreement, training gate — `configs/risk_events.yaml`, `src/risk/labels.py`, `scripts/validate_risk_labels.py`.
35. Event-annotation UI — `src/ui/annotate_events.py`.
36. Removed bad targets (gallbladder made every frame "Medium"; cystic duct failed on unseen data).

## 4. Pipeline, app, performance

37. End-to-end pipeline + overlay video, CSV/JSON export — `src/pipeline/{pipeline,overlay}.py`, `run_pipeline.py`.
38. Streamlit app — `src/ui/app.py` (now shows tested/untested target badges, collapsible notices).
39. Live dashboard — `src/ui/training_monitor.py` (system, watchdog, jobs, ETA, TipNet, results).
40. CPU profiling + fast preset `configs/pipeline_cpu_fast.yaml` (3.2 → ~7 fps; 80–100 % same risk category).
41. Resource watchdog — `scripts/watchdog.py` (memory trim, throttle, battery suspend).
42. Video helpers — `scripts/make_video_from_frames.py`, `make_synthetic_video.py`.

## 5. External data and experiments

43. CholecSeg8k: Kaggle download of all 17 videos (32,320 files), duplicate check — `scripts/compare_datasets.py`.
44. EndoVis'15 robotic: label/video validation, DLC conversion, 79-epoch GPU training — `src/external/endovis.py`,
    `scripts/endovis_{validate,to_dlc,evaluate}.py`; held-out Dataset4 recall 81 %.
45. EndoVis official TEST videos found on a public mirror and downloaded; filename-pattern bug fixed; 910/910 test
    frames aligned; official test: **median 3.2 px**, recall 57 % (79–92 % on seq 1–4) — `scripts/endovis_evaluate_test.py`.
46. Endoscapes (6.3 GB, cystic duct masks) downloaded — integration pending.
47. Cholec80 checked (75 GB, no masks) — remote per-video access verified, not downloaded.

## 6. Security, reliability, docs

48. Kaggle token revoked in your account and verified (CLI rejected); local copy deleted; `.gitignore` hardened.
49. Bugs fixed along the way: DLC CSV index parsing, CSV schema loss, proximity cache id() bug, class_support group_by,
    DLC snapshot loading on DirectML (`scripts/dlc_snapshots_to_cpu.py`, torch.load patch), dashboard ETA, dashboard
    crash on TipNet history, pandas `null`/`.clip`/`.flags` pitfalls.
50. Tests: 9 → **16 test files, 104 passing**; legacy Dataset_Utilities 3 pass / 2 skip (no fabricated data).
51. Docs: README, PROJECT_STATUS_REPORT, PROJECT_AUDIT, DATASETS, DLC_ANNOTATION, RISK_ANNOTATION,
    ENDOVIS_INTEGRATION, WORK_LOG, this file.

## 7. Open items

* TipNet: retry with better initialisation (in progress).
* Endoscapes: index and external tissue/cystic-duct evaluation.
* Final report update (TASK 11).
* Needs you/experts: risk-event labels, clinical confirmation of event definitions, calibration for mm.

## 8. Comparison table: before vs now

| # | Area | Before (earlier work) | Now (our work) | Evidence |
|---|---|---|---|---|
| 1 | Instrument segmentation | U-Net, wrong target, Dice 0.26–0.32 | ResNet-18 U-Net, grasper 0.852 / L-hook 0.840 on unseen patients | outputs/models/*_multivideo/eval_* |
| 2 | Tissue segmentation | none | 7 tissue classes, mean Dice 0.735 on unseen patients | tissue_seg_multivideo/eval_test |
| 3 | Training data | 1 video | 17 videos, patient-level split, duplicate-checked | outputs/splits/video_splits.yaml |
| 4 | Class reliability | none | tested / barely tested / untested flags per tissue | class_support.json |
| 5 | Temporal stability | none | EMA smoothing, flicker −21 % / −12 % | outputs/eval_temporal |
| 6 | Instrument tip tracking | none | mask-tip + 2-grasper tracking | src/tracking |
| 7 | Tip accuracy vs human labels | none | L-hook 22.4 px, grasper 128 → 31.9 px | outputs/logs/tipeval_* |
| 8 | Tip labels | none | 40 human (yours) + 6,800 automatic pseudo-labels (16 videos) | outputs/auto_labels |
| 9 | Deep keypoint models | none | DLC ×3 experiments (rejected), TipNet (in progress) | docs/DLC_ANNOTATION.md |
| 10 | Motion analysis | none | speed, direction, gap reset, camera motion | src/motion |
| 11 | Proximity | none | pixel distance to tissue with safety flags | src/proximity |
| 12 | Risk scoring | none | rule-based score with explanations; learned model gated | src/risk |
| 13 | Risk-event labelling | none | schema, validator, agreement, annotation app | src/risk/labels.py, annotate_events.py |
| 14 | Pipeline | notebooks only | end-to-end video → annotated video + CSV/JSON | run_pipeline.py |
| 15 | User interface | none | main app, dashboard, tip-label tool, event tool | src/ui |
| 16 | CPU speed | not measured | 3.2 fps default, ~7 fps fast preset (not real-time) | outputs/bench |
| 17 | GPU use | none | AMD RX 6800M via DirectML for all training | .venv-dml |
| 18 | Robotic keypoints (EndoVis) | none | official test median 3.2 px, recall 57 % (79–92 % seq 1–4) | outputs/endovis/eval_official_test |
| 19 | External datasets | none | CholecSeg8k 17 videos, EndoVis train+test, Endoscapes 6.3 GB | data/ |
| 20 | Tests | 5 (2 failing) | 104 passing (+ legacy 3 pass / 2 skip) | tests/ |
| 21 | Documentation | 1 README | 9 docs incl. audit, status report, work log | docs/ |
| 22 | Security | API token exposed in chat | token revoked + verified, secrets git-ignored | WORK_LOG TASK 0 |
| 23 | Laptop safety | none | watchdog: heat/memory/battery throttling | scripts/watchdog.py |
