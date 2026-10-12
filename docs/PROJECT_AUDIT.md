# PRISM-Endo audit checklist (2026-10-10)

Evidence and numbers: [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md). Baseline tests before changes: 86 passed;
after changes: 87 passed.

| # | Item | Status |
|---|---|---|
| 1 | Instrument segmentation (grasper) on unseen patients | Completed and verified (Dice 0.852) |
| 2 | Instrument segmentation (L-hook) on unseen patients | Completed and verified on unseen val videos (Dice 0.840; slight selection bias) |
| 3 | Tissue segmentation on unseen patients | Completed and verified (mean Dice 0.735; per-class in the report) |
| 4 | Cystic duct / hepatic vein / liver ligament as reliable targets | Blocked by missing data (no adequate held-out frames) |
| 5 | DeepLabCut integration (I/O, gates, workflow) | Completed and verified (tests + synthetic end-to-end run) |
| 6 | Laparoscopic tip labels | Partially completed: 40/40 eval frames human-labelled (4 points flagged); 0/200 training |
| 7 | Laparoscopic DLC training and tip accuracy | Blocked by item 6 |
| 8 | Mask-tip fallback, multi-instance tracking | L-hook measured vs human labels: median 22.4 px; grasper pending a definition |
| 9 | Motion / trajectory / camera motion | Completed and verified (unit tests; camera fit measured) |
| 10 | Temporal smoothing | Completed and verified (flicker −21 % tissue, −12 % instrument on video01 test clips) |
| 11 | Proximity (px), reliability flags | Completed and verified (tests; unavailable when tissue/keypoint missing) |
| 12 | Rule-based risk | Implemented but not fully validated (uncalibrated heuristic) |
| 13 | Learned risk, event metrics | Blocked: no expert labels (schema, validator and UI done) |
| 14 | 17-video dataset, video-level split, duplicate check | Completed and verified |
| 15 | EndoVis label/video validation and DLC conversion | Completed and verified (940/940 matched, 3 ambiguous excluded) |
| 16 | EndoVis DLC training and evaluation | Completed and verified as a separate robotic experiment (79 epochs; held-out recall 81%, median 3.8 px) |
| 17 | EndoVis official test videos | Blocked: registration/download needed |
| 18 | CPU performance + ONNX | Completed and verified (3.37 → 7.67 fps on unseen video17) |
| 19 | Fast-preset accuracy | Completed and verified (3 unseen clips: 80–100 % same risk category, distance diff < 1 px) |
| 20 | Streamlit app, labelling, annotation, monitor UIs | Completed and verified (headless tests) |
| 21 | Legacy notebook model | Verified obsolete (wrong target); kept, unused |
| 22 | Docs: DATASETS, DLC_ANNOTATION, RISK_ANNOTATION, ENDOVIS_INTEGRATION, this audit | Completed |
| 23 | Tissue model epoch 12 | Completed (no improvement over epoch 8) |
| 25 | Learned risk model | Not done by design: needs real event labels |

## Re-verification (2026-10-10, ~17:20, before the autonomous completion round)

Checked against files rather than docs:
* Tests: `pytest tests` → **87 passed**. `pytest Dataset_Utilities` → 3 passed, **2 failed** (FileNotFoundError: the
  legacy `sample_tool_presence.csv` / `sample_risk_annotations.json` are not in the folder; not project code).
* Checkpoints present: instrument_seg, instrument_seg_multivideo, tissue_seg, tissue_seg_multivideo (`best.pt` + `model.onnx`).
* Re-read metrics: instrument multivideo test (videos 17/25/28) grasper Dice 0.852, no L-hook in test; val (20/26/43)
  grasper 0.918, L-hook 0.840; tissue multivideo test mean 0.735 (cystic duct 0.0, too few test pixels).
* External data present: EndoVisPoseAnnotation (4 MB), Tracking_Robotic_Training.zip (39 MB),
  Segmentation_Robotic_Training.zip (40 MB), extracted tracking folder.
* Updates since the table above: Kaggle token revoked and verified (WORK_LOG.md, TASK 0).
