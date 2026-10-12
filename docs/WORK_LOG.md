# PRISM-Endo work log

Resume rule: continue from the first task not marked DONE. One heavy GPU/CPU training job at a time.

## TASK 0 — Kaggle token (DONE, verified 2026-10-10 ~17:15)
- Kaggle Settings → API Tokens (browser already signed in): the only token, "prism" (created ~15 h earlier), was
  expired via its menu → Expire Token → confirm.
- Verification: (1) after a reload the token table is empty; (2) the Kaggle CLI using the old local token now returns
  "Authentication required". No legacy kaggle.json key existed locally.
- Local copies: ~/.kaggle/access_token (now dead) deleted. Token pattern not found in any project file or in git
  history (`git log --all -p`). .gitignore now excludes .kaggle/, kaggle.json, access_token, *.token, .env.*
- Remaining exposure: a local chat transcript still contains the (now revoked)
  value. It is harmless after revocation; delete that session file if desired.
- No replacement token created (all needed data is already downloaded).

## TASK 1 — Audit baseline (DONE)
Re-verification appended to docs/PROJECT_AUDIT.md: 87/87 project tests pass; 2 legacy Dataset_Utilities tests fail on
missing sample files; checkpoints and external data present; metrics re-read from metrics.json files.

## TASK 2 — Automatic labelling of the 200 training frames (DONE)
- New: src/tracking/auto_tips.py (geodesic far end; grasper hinge = jaw_px back along the instrument),
  scripts/auto_label_tips.py -> outputs/auto_labels/{train_auto_labels.csv, summary.json, contact_sheet_1-5.jpg},
  DLC project prism-tips-AUTO-* (scorer auto_geo_v1). PSEUDO labels, not human-verified.
- Grasper definition: jaw HINGE (the convention of the 40 human eval labels). jaw_px=90 tuned on the TRAINING side
  (sweep 20..140 px; alpha*shaft-width variant worse). Human eval frames not used for tuning.
- Labeller accuracy vs the 40 human frames (GT masks; measured once after freezing):
  L-hook median 11.7 px (p90 25, PCK@20 0.78), grasper hinge median 17.7 px (PCK@20 0.63, p90 387 = fragment failures).
  Previous: grasper 108-129 px (jaw-end), L-hook 22 px (pipeline, predicted masks).
- 200 frames: 118 L-hook, 108 grasper, 2 second-grasper points; points flagged for review (incl. 35 low
  confidence, 16 temporal jumps >80 px). Every point has confidence, method and disagreement fields.

## TASK 3 — Keypoint method improvement + DLC on auto labels (IN PROGRESS)
- Leakage check: instrument_seg_multivideo was trained on ALL of video01, which contains the 4 human-evaluated clips.
  Tip comparisons therefore use the single-video `instrument_seg` (those clips are in its held-out test split).
- New default (configs/pipeline.yaml keypoints.mask_tip_landmarks: grasper hinge, jaw_px 90, half-res geodesic):
  vs 40 human frames, predicted masks, min-conf 0: grasper median 128.3 -> 31.8 px (p90 143.8 -> 48.9, PCK@20 0 -> 0.27);
  L-hook unchanged 22.4 px (geodesic variant 23.4 px, not adopted). Keypoint cost 15 -> 45 ms/frame;
  pipeline 5.4 fps on these clips (instrument-only). Runs: outputs/runs_tips_{old,geo,hinge}; logs
  outputs/logs/tipeval_runs_tips_*.json. tests/test_auto_tips.py (6 tests) added.
- DLC on auto labels: scripts/run_auto_label_dlc.sh (GPU 60 ep, final-epoch snapshot, eval at conf 0.5 and 0.1),
  log outputs/logs/auto_dlc.log. If interrupted: check "all done" in the log; else rerun the script (it skips
  dataset creation; DLC restarts epoch numbering - back up the train folder first).

## TASK 4 — EndoVis audit (DONE in the earlier round; re-checked)
Validation report, gallery overlays (outputs/endovis/gallery), DLC conversion, held-out Dataset4 evaluation exist;
segmentation zip videos are byte-identical to the tracking ones. Blocker: official EndoVis'15 test videos need
registration on the challenge site (not public).

## TASK 5 — Dataset expansion (IN PROGRESS)
- Endoscapes (CAMMA, CC BY-NC-SA 4.0, direct public link, 6.29 GB): Seg50 has gallbladder, CYSTIC DUCT, cystic
  artery, cystic plate, dissection, tool masks (493 frames / 50 videos). Downloading to data/endoscapes/.
- Cholec80 (74.9 GB): phase + tool-presence only, no masks/keypoints -> not downloaded (no supervised use here).
- CholecT50: triplets (+ boxes on 5 videos) -> not useful for masks/keypoints. PhaKIR: restricted, not attempted.

## TASK 10 — partial
Dataset_Utilities: the 2 failures need sample files never shipped (folder is git-ignored legacy). Tests now SKIP
with a reason when the file is absent (no data fabricated; original kept in the job tmp). Result: 3 passed, 2 skipped.

### Interruption (memory pressure stopped background jobs)
- DLC on auto labels stopped at epoch 30/60 (snapshots 005-025 + snapshot-best-030 saved; valid loss 0.00548).
  Resume: back up the train folder, convert snapshots (dlc_snapshots_to_cpu), train 30 more epochs with
  --snapshot snapshot-best-030.pt, then the analyze/eval steps of scripts/run_auto_label_dlc.sh.
- Endoscapes download stopped at 3.28/6.29 GB; resume with `curl -C -` (same command).
- Dashboard (8600) and watchdog also stopped. Not restarted automatically (waiting for the user's go-ahead).

### TASK 3 result
The interruption notice only stopped the shell wrapper: training finished all 60 epochs (3257 s) and the chain ended
"all done 18:29". DLC on auto labels (final epoch, conf 0.5): L-hook 0/37, grasper 23/49 median 40.7 px p90 508 px ->
NOT adopted. Default = mask-based (grasper hinge 31.8 px, L-hook 22.4 px). Documented in DLC_ANNOTATION.md.

## TASK 9 — CPU benchmark after the keypoint change (DONE)
Same unseen clip as the original baseline (data/cholecseg8k/video17/video17_01563, 80 frames, CPU, ONNX in fast preset):
| config | fps | ms/frame (inst / kp / tissue) | earlier baseline |
|---|---|---|---|
| default (pipeline.yaml) | 3.24 | 301 (122 / 18 / 140) | 3.37 |
| fast (pipeline_cpu_fast.yaml) | 6.8–7.2 | 131 (56 / 22 / 31) | 7.67 |
The Endoscapes download ran in parallel, so ~5% noise. Not real-time.
Regression found and fixed: the hinge method gave confidence ~0.09 on video17, where the grasper hugs the FOV border
(visible length 12–30 px < 90 px jaw), so risk was "Unavailable" in 57/80 frames. Fix: fall back to the straight-line
far end when the visible length is < jaw_px+10 (method tagged far_end_fallback). After: video17 10/80 unavailable
(old rule 11/80); human frames grasper median 31.9 px (p90 93), L-hook 22.4 px. Tests: 94 passed.

## TASK 7 — Trajectory / camera motion / proximity (DONE: verified existing; no change warranted)
EMA smoothing (alpha 0.7) with gap reset (max_gap_s) so abrupt real moves after a gap are not smoothed away; Hungarian
slot tracking for 2 graspers; camera motion (LK+RANSAC) is MEASURED and logged (`*_comp` speed columns), smoothing
compensation measured with no benefit -> off; `use_camera_compensated_speed: false` for risk because there is no ground
truth to validate it. Proximity in px with FOV-border exclusion, unavailable when tissue/keypoint is missing or
low-confidence. All distances/speeds in image units. Covered by test_motion/test_tracking/test_proximity.
New this round: the grasper keypoint landmark (TASK 3) feeds proximity; its fallback keeps risk availability
(video17: 70/80 frames ok).

## TASK 8 — Risk scoring and event workflow (DONE as far as labels allow)
Rule-based mode operational (test_risk: 12 tests). Event-annotation UI (src/ui/annotate_events.py), schema
(configs/risk_events.yaml, still PROPOSED definitions), validator and training gate exist. New: tests/test_risk_labels.py
(6 tests: annotation round-trip, interval maths, kappa, validator rejects bad rows/columns, gate refuses without
labels). Learned risk model NOT trained: no expert event labels exist (rule-generated labels would be circular).

## TASK 10 — Tests and UI reliability (DONE)
New tests: test_auto_tips.py (7), test_risk_labels.py (6), test_ui_smoke.py (4: training_monitor, app, annotate_events,
label_tips render headlessly via streamlit AppTest). Full run: `pytest tests` -> 104 passed. Dataset_Utilities -> 3
passed, 2 skipped (missing sample files).

## 2026-10-11 evening
- **Clean DLC evaluation done** (docs/DLC_CLEAN_EVALUATION.md): video52/55 fully excluded from training; DLC-only
  6.5 px median, 35/35 found, 91 % within 20 px on 35 human clicks (earlier leaky fold model: 9.8 px).
- **Tissue + DSAD** (Dresden Surgical Anatomy Dataset, CC BY 4.0): fine-tune on Colab T4 (scripts/train_tissue_dsad.py,
  pack via scripts/make_colab_tissue_pack.py). DSAD test 0.380 -> 0.846, but CholecSeg8k full test 0.735 -> 0.697,
  so NOT adopted. Appendicectomy video: "liver" pixels 9.9 % -> 4.6 % (scripts/compare_tissue_on_video.py).
  Guarded local retrain queued (--max-cholec-drop 0.01, --dsad-weight 0.5, lr 5e-5): outputs/models/tissue_seg_dsad_local.
- Watchdog: GPU cool-down duty cycle (240 s run / 45 s pause, doubled when throttled).
- Full test suite: 112 passed.
- **Guarded local retrain finished (22:3x)**: full CholecSeg8k + DSAD, 8 epochs. CholecSeg8k val never came within 0.01
  of the original (best 0.883 vs 0.907), so the guard was never met. Test: CholecSeg8k 0.735 -> 0.696, DSAD 0.380 -> 0.844.
  NOT adopted. Conclusion: one 12-class model cannot keep gallbladder quality and fix other surgeries at once;
  per-surgery model selection is the practical route. Default tissue model unchanged.
- **App "Surgery type" switch** (configs/pipeline.yaml tissue.by_surgery): Gallbladder -> tissue_seg_multivideo,
  Other abdominal -> tissue_seg_dsad_colab (class_support.json + eval_test written; CholecSeg8k test 0.697).
- **Bleeding detector in pipeline + app** (src/safety/bleeding.py, safety.bleeding): columns bleeding_frac /
  bleeding_detected, thin red outline in the overlay, "Bleeding flagged" metric + chart in the app.
  Check: held-out SurgBlood clips supple6 26/27 and supple9 105/106 bleeding frames flagged; appendicectomy video
  0/1122 flagged (no visible bleeding). Tests: 114 passed (tests/test_bleeding.py added).

## 2026-10-12 React UI (draft on :8510)
- web/ (React 18 + Vite + Tailwind 4 + Geist) served by src/api/server.py (FastAPI): runs, videos, frames, flags,
  upload, start/stop analysis (temp folder, finalised only on success), active-run discovery.
- Fixes from testing: playback re-render freeze (memoised charts/timeline, 10 Hz playhead, debounced inspector frame),
  static-file path traversal, native <select> white popup in dark mode (themed Select component), dialog open/close
  race, two-line "Why this risk" box, tip source default -> hybrid (DLC was skipped), DLC-only auto-analyse.
- E2E: web/e2e/app.spec.ts (Playwright + axe): 15 tests (every dropdown light/dark, header buttons, timeline,
  signals, events, flags, inspector + shortcuts, upload, start/stop, auto-open, mobile, WCAG 2 AA scan);
  30/30 passed over two repeats. Run: cd web && npm run test:e2e

## 2026-10-12 later
- Sidebar redesign (numbered groups), tissue annotation fields (targets with reliability, outline all, overlay
  strength) wired through `/api/run` -> pipeline config; grey light theme; layout fixes (full height, no wrapped
  buttons); themed Select replaces native <select>; dialog open/close race fixed.
- Speed: predictor / CVS / bleeding models cached across pipelines; DLC first-run analysis in a subprocess so Stop
  ends it; phases reported to the UI ("Running DeepLabCut…", "Converting videos…"); Windows path display fixed.
- Playwright suite: 20 tests, all passing; afterEach stops any analysis a test left running.
- Note: a failing test run re-analysed appendicectomy_max fully (1,124 frames) — the user's 200-frame run was replaced.
- Public link: Cloudflare quick tunnel (temporary URL, full access, chosen by the user). Antigravity cannot forward
  ports (code-tunnel.exe missing). Permanent URL pending: needs the user's devtunnel / ngrok / Cloudflare account.
- Nothing committed (user instruction).
