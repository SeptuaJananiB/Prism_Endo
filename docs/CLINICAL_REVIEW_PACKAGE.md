# PRISM-Endo — clinical review package (for a supervising surgeon)

PRISM-Endo is a **research prototype**. Nothing in it is clinically validated. This package lists every decision that
must be confirmed or changed by a surgeon before the risk outputs can be interpreted clinically. The reviewer fills in
`docs/clinical_review_form.html` (opens in any browser, saves a `clinical_review_<name>.json`), which then replaces the
engineering proposals.

## 1. What the system shows (and the evidence behind each output)

| Output | How it is produced | Evidence (held-out data) | Not established |
|---|---|---|---|
| Instrument masks | U-Net (CholecSeg8k, 17 videos) | grasper Dice 0.852, L-hook 0.840 | other instruments (clipper, scissors…) |
| Tissue masks | U-Net (CholecSeg8k) | mean Dice 0.735; liver/fat/gallbladder tested | cystic duct, hepatic vein, liver ligament: NOT reliable |
| L-hook tip | DeepLabCut v3 (86 human clicks) + geometry | median 18.4 px on 40 human frames | — |
| Grasper point (jaw hinge) | geometry | median 31.9 px on 40 human frames | other videos (labels pending) |
| Tip-to-tissue distance | pixels; approx. mm from the 5 mm shaft width | — | mm accuracy (no camera calibration; ±30–50 %) |
| Risk score / category | rule-based formula (section 2) | none — never compared with surgeon judgement | everything clinical |
| Critical View of Safety | ResNet-50 on Endoscapes surgeon labels | test mAP 0.539 (40 videos) | use as a decision aid |
| Bleeding (in preparation) | U-Net on SurgBlood surgeon-reviewed masks | see bleeding_report.json | transfer to these videos |

## 2. Rule-based risk score — parameters to confirm (`configs/pipeline.yaml`, section `risk`)

Score = weighted sum of four terms, smoothed over time, then mapped to a category.

| Parameter | Current value | Meaning | Question for the surgeon |
|---|---|---|---|
| proximity_ref_frac | 0.08 of image diagonal (≈78 px, very roughly 5–10 mm) | distance at which the proximity term becomes 0 | At what distance should concern start? |
| speed_ref_frac_per_s | 0.6 diagonals/s | speed at which the speed term saturates | Which tip speeds near tissue are unsafe? |
| approach_ref_frac_per_s | 0.3 diagonals/s | approach speed towards tissue that saturates | — |
| accel_ref_frac_per_s2 | 12 diagonals/s² | abrupt-movement saturation | — |
| weights | proximity 0.5, approach 0.2, speed 0.15, abrupt 0.15 | relative importance | Is proximity the dominant factor? |
| thresholds | Low 0.25, Medium 0.5, High 0.75 | category boundaries | Acceptable false-alarm rate per minute? |
| dwell_s / rapid_rise_per_s | 0.2 s / 2.0 per s | debounce vs. immediate escalation | — |
| proximity targets | liver (only tested target) | which tissue proximity matters | Which structures must be targets (cystic duct, artery, CBD…)? |

## 3. Risk-event definitions to confirm (`configs/risk_events.yaml`, currently `status: proposal`)

near_miss_critical_structure · unintended_contact · thermal_spread_risk · bleeding_onset · unsafe_traction · other ·
review_complete (full text in the YAML). For each: accept / rewrite / remove, severity scale, and what counts as
"dangerously close". After confirmation the YAML `status` becomes `confirmed`; only then can expert-labelled events
train the learned injury-risk model (`scripts/train_risk_model.py`, gated by `scripts/validate_risk_labels.py`).

## 4. How to review

1. Open the app (`.venv\Scripts\streamlit run src/ui/app.py`), choose *Previous run*, watch 2–3 clips with the overlay
   (risk banner, tip markers, CVS line) and the frame inspector.
2. Fill `docs/clinical_review_form.html` and press **Save** (downloads a JSON file); send it back to the developer.
3. Optional: label risk events with the annotation tool (`src/ui/annotate_events.py`) on a few videos.

Sign-off by a qualified surgeon is required; the developer/AI cannot sign off.
