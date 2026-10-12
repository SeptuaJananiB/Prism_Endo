PRISM-Endo — acknowledgement of work completed (status 2026-10-11, 13:30)

Your task list has been executed. Results below are measured on held-out data only; nothing is fabricated.

1. SECURITY (Task 0)
- Exposed Kaggle API token revoked in the account (Settings > API) and verified: token list empty, CLI rejects the old token. Local copy deleted, secrets git-ignored.

2. AUDIT (Task 1)
- Project re-verified against files and test runs (docs/PROJECT_AUDIT.md).

3. AUTOMATIC LABELLING + DEEPLABCUT (Tasks 2-3, main goal)
- Grasper landmark fixed as the jaw hinge (user convention). Automatic labeller (auto_geo_v1) made 6,800 pseudo-labels from 16 videos (video01 excluded).
- Finding: CholecSeg8k L-hook masks often omit the thin metal hook, so automatic L-hook labels stop 20-50 px short of the true tip.
- User clicked 86 L-hook tips (+10 "not visible") on 6 other videos. DLC v3 trained on those human clicks (AMD GPU).
- New app default = HYBRID tip source (DLC v3 for the L-hook where plausible, geometry otherwise):
  * 40 human-labelled video01 frames: L-hook 22.4 -> 18.4 px median; grasper hinge 31.9 px (unchanged).
  * Wider test, video52+55 held out (35 human clicks): DLC/hybrid 9.8 px, 35/35 found vs geometry 28.6 px, 25/35. Caveat: the starting model had seen those videos with automatic labels, so this is optimistic.
- Earlier DLC attempts on machine/AI/automatic labels were tested and rejected (documented).
- The pipeline runs DLC automatically for a new video (cached).

4. ENDOVIS (Task 4)
- Official EndoVis'15 robotic test videos found on a public mirror; label-format bug fixed; 910/910 frames aligned. Official test: median 3.2 px, recall 57 % (79-92 % on sequences 1-4). Kept as a separate robotic experiment.

5. DATASETS (Task 5)
- CholecSeg8k (17 videos), EndoVis train+test, Endoscapes (6.3 GB) integrated. Cholec80 assessed (no masks). Risk-event datasets checked: IEEE iAE has no files, SAGES CVS not public, CholecTrack20 needs the user's data-use agreement.

6. SEGMENTATION (Task 6)
- Instrument: grasper Dice 0.852, L-hook 0.840; tissue mean Dice 0.735 on unseen patients; untested tissue classes flagged in the app.

7. MOTION / PROXIMITY (Task 7)
- Verified; demonstrated live (tip-speed charts; tracking spikes identified and explained).

8. RISK / SAFETY (Task 8)
- Rule-based risk score kept. New learned SAFETY model: Critical View of Safety (Endoscapes surgeon labels), ResNet-50 trained on a free Google Colab T4 (192 -> 60 s/epoch after RAM caching + fp16). Final (v2, chosen on validation): test mAP 0.539 on 40 unseen videos, re-verified on the laptop. Integrated into the pipeline, video overlay and app. It is a "safe to clip?" estimate, not tissue-injury risk.

9. PERFORMANCE (Task 9)
- CPU: ~3.2 fps default, ~7 fps fast preset (not real-time).

10. TESTS / UI (Task 10)
- 110 automated tests pass (legacy folder: 3 pass, 2 skipped for missing sample files, no data fabricated). App, dashboard, labelling tools and a readable status board (status/STATUS_BOARD.html + Excel) work.

11. DOCUMENTATION (Task 11)
- README (final results), PROJECT_STATUS_REPORT, DLC_ANNOTATION, ALL_WORK_DONE (before/after comparison), WORK_LOG, status board.

STILL OPEN (needs outside input)
- Learned tissue-injury risk model: needs surgeon-labelled injury/near-miss events.
- Bleeding detection: needs the CholecTrack20 data-use agreement signed by the user.
- Clinical sign-off of risk rules; distances in mm need camera calibration.
- Optional: fully clean wider DLC test; label 24 skipped L-hook frames.
- Git: nothing committed since the initial commit (by the user's choice).
