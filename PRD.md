# Product Requirements — PRISM-Endo

> Research prototype. This document describes product intent and acceptance criteria for a research tool;
> it is not a regulatory specification.

## 1. Problem

During laparoscopic surgery, injuries happen when instruments come too close to critical structures, move
abruptly, or when the surgeon clips before the Critical View of Safety (CVS) is reached. Reviewing surgical video
for these moments is slow and subjective. There is no simple tool that marks *where the instruments are, what they
are close to, how they move, and whether bleeding or CVS is visible* — frame by frame — with honest reliability
information.

## 2. Goals

| # | Goal | Measure |
|---|---|---|
| G1 | Track instruments and their tips in every frame | Dice on unseen patients; tip error in px vs human clicks |
| G2 | Measure tip-to-tissue proximity for chosen targets | per-class tissue Dice; "tested / not tested" flag per class |
| G3 | Explain risk moments in plain language | every risk value has a written reason |
| G4 | Detect bleeding and estimate CVS | F1 / AUROC (bleeding), mAP (CVS) on unseen data |
| G5 | Make review fast | find any risk event in ≤ 2 clicks or one key press |
| G6 | Be honest | no metric reported without held-out evaluation; leaks documented |

Non-goals: clinical decision support, real-time intra-operative alerts, millimetre-accurate measurement without
calibration, any claim of injury probability.

## 3. Users

| User | Need |
|---|---|
| **Surgical researcher** | analyse videos, export per-frame data, compare models |
| **Surgeon reviewer** | watch a case, see why risk rose, flag where the AI is wrong |
| **ML engineer** | retrain components, run evaluations, extend the pipeline |

## 4. Scope and requirements

### 4.1 Pipeline

| ID | Requirement | Status |
|---|---|---|
| P1 | Instrument masks for grasper and L-hook | done — Dice 0.852 / 0.840 |
| P2 | Tip keypoints: DeepLabCut for L-hook, geometry for grasper (hybrid), DLC run once per video and cached | done — L-hook 6.5 px (clean test) |
| P3 | Tissue masks, 12 classes, per-class reliability from held-out tests | done — mean Dice 0.735 |
| P4 | Surgery-type tissue model selection (gallbladder / other) | done |
| P5 | Tip motion: speed, direction, abrupt change, camera-motion compensation | done |
| P6 | Proximity to user-chosen target tissues; warning for untested targets | done |
| P7 | Rule-based risk score with per-frame explanation | done |
| P8 | Bleeding detection with overlay | done — F1 0.937 |
| P9 | Critical View of Safety estimate | done — mAP 0.539 |
| P10 | Outputs: annotated video, per-frame CSV, run summary with model provenance | done |
| P11 | Approximate mm from instrument shaft width, clearly marked | done (±30–50 %) |

### 4.2 Web app

| ID | Requirement | Status |
|---|---|---|
| W1 | Upload or choose a video, set options, start analysis | done |
| W2 | Live progress with phase text; Stop that takes effect within 1 s and never corrupts earlier results | done (~0.2 s) |
| W3 | Finished analysis opens automatically and is highlighted | done |
| W4 | Synced video + stacked timeline + charts; click anywhere to jump | done |
| W5 | Events list, milestones, plain-language summary, printable report | done |
| W6 | Reviewer flags saved with the analysis | done |
| W7 | Compare two analyses | done |
| W8 | Keyboard control, light/dark theme, WCAG 2 AA | done (axe scan clean) |
| W9 | Shareable link | temporary tunnel done; permanent URL needs an account |

## 5. Success metrics (current)

- All component metrics above measured on held-out patients or clips.
- 117 unit tests and 20 end-to-end browser tests pass.
- Settings change → analysis starts in 0.3 s; Stop → 0.2 s.

## 6. Open items

1. Clinical sign-off by a surgeon ([docs/CLINICAL_REVIEW_PACKAGE.md](docs/CLINICAL_REVIEW_PACKAGE.md)).
2. Learned grasper-tip model (needs ~100 human jaw-hinge clicks).
3. Calibrated millimetres.
4. Learned injury-risk model (needs expert event labels).
5. Real-time speed (currently ~5–6 fps).
6. Permanent public URL.
