# Gap Analysis — Surgical Risk Prediction in Laparoscopic Video

This document identifies the gaps in current surgical risk prediction from endoscopic video, shows which ones
PRISM-Endo addresses, and lists what remains open.

## 1. Background

Laparoscopic surgery is performed through a camera, so most information about safety is visible in the video:
instrument positions and motion, the anatomy being worked on, bleeding, and whether the Critical View of Safety (CVS)
has been reached before structures are divided. Research has produced strong building blocks — surgical phase
recognition, instrument segmentation, tool tracking, anatomy segmentation and CVS assessment — but they are usually
studied **separately**, evaluated on **frames rather than patients**, and rarely connected to a **risk signal that a
reviewer can act on**.

## 2. Gaps identified

| # | Gap in current practice / research | Why it matters |
|---|---|---|
| G1 | **Isolated tasks** — instruments, anatomy, bleeding and CVS are modelled separately | Risk comes from their *interaction* (a tool, moving fast, close to a structure), which no single model captures |
| G2 | **No tool-to-tissue proximity signal** | Most systems find instruments or anatomy, not *how close* a tool tip is to a specific structure over time |
| G3 | **Motion is ignored** | Speed, approach towards tissue and abrupt changes are precursors of injury but are rarely quantified |
| G4 | **Black-box outputs** | A score without a reason cannot be checked by a surgeon or used for teaching |
| G5 | **Unclear reliability** | Models seldom say which structures they were actually validated on; rare structures (cystic duct) are often unreliable but not flagged |
| G6 | **Frame-level evaluation and leakage** | Splitting by frame lets near-identical frames from the same patient appear in train and test, overstating accuracy |
| G7 | **No early warning** | Systems describe the current frame; few estimate *time to contact* before it happens |
| G8 | **Offline only** | Many pipelines process recorded files and cannot work from a live camera |
| G9 | **No labelled injury / near-miss data** | Learned *injury prediction* needs expert-labelled events that public datasets do not provide |
| G10 | **Hard to review** | Long videos without timelines, events or summaries make review slow |
| G11 | **Domain shift** | Models trained on one operation (gallbladder) mislabel anatomy in other operations |

## 3. How PRISM-Endo addresses them

| Gap | PRISM-Endo | Status |
|---|---|---|
| G1 | One pipeline combining instruments, tips, motion, tissue, proximity, bleeding and CVS | ✅ Addressed |
| G2 | Per-tip distance to user-selected tissues, every frame | ✅ Addressed |
| G3 | Smoothed tracking: speed, approach speed, acceleration, abrupt changes, camera motion | ✅ Addressed |
| G4 | Rule-based risk with a written reason per frame + gpt-oss summary of the whole video | ✅ Addressed |
| G5 | Every tissue marked tested / barely tested / not tested on unseen patients; low-confidence tips flagged | ✅ Addressed |
| G6 | Patient-level splits, held-out-only reporting, leakage audit and a leakage-free re-evaluation | ✅ Addressed |
| G7 | **Time-to-contact early warning** (distance ÷ approach speed) with on-video alert, events and timeline track | ✅ Addressed (estimate, not validated) |
| G8 | **Live capture** from a webcam / capture card / stream URL with OpenCV, live annotated view in the browser | ✅ Addressed (30 fps per-frame tracking + keyframe models) |
| G9 | Reviewer flags collect expert agreement and corrections as the start of a labelled dataset | ⚠️ Started |
| G10 | Synced timeline, events, milestones, compare, printable summary, read-aloud explanation | ✅ Addressed |
| G11 | Surgery-type selection with a second tissue model trained on other operations (DSAD) | ⚠️ Partly addressed |

## 4. What remains open

1. **Validated injury prediction** — requires expert-labelled injury and near-miss events (G9); the current risk
   score and time-to-contact warning are research estimates.
2. **Critical structures** — cystic duct, cystic artery and common bile duct need more labelled data before any
   guidance about them is possible.
3. **Full models on every frame** — live mode reaches 30 fps by tracking between keyframes (~4–6 fps); running
   every model on every frame would need model optimisation (smaller models, ONNX / TensorRT).
4. **Calibrated measurement** — true millimetres need camera calibration or depth estimation.
5. **Clinical validation** — prospective review by surgeons ([CLINICAL_REVIEW_PACKAGE.md](CLINICAL_REVIEW_PACKAGE.md)).

## 5. Objectives check

| Objective | Status |
|---|---|
| Identify gaps in surgical risk prediction | ✅ This document |
| Extract, track and analyse tool motion and tissue proximity | ✅ Pipeline stages 1–5 |
| Predict pre-injury risk and generate visual warnings | ✅ Risk score + time-to-contact warning, on-video alerts, timeline and events (research estimate) |
| Real-time endoscopic video capture using OpenCV | ✅ Live camera / capture card / stream capture, 30 fps per-frame analysis |
