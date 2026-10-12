# PRISM-Endo — Technical guide (full commands, measurements and history)

> **Research prototype. Not a clinically validated surgical safety system.**
> All distances and speeds are **image-space pixels** (px, px/s), not millimetres.
> The risk score is a **rule-based heuristic research score**, not a probability of injury.

> **Status (2026-10-12):**
>
> | Component | Result (held-out data only) |
> |---|---|
> | Instrument segmentation | grasper Dice 0.852, L-hook 0.840 on unseen patients |
> | Tissue segmentation | mean Dice 0.735 on unseen patients (cystic duct / hepatic vein / liver ligament flagged untested) |
> | **Tip keypoints (DeepLabCut, default `keypoints.source: hybrid`)** | **L-hook 18.4 px** median on the 40 human-labelled video01 frames (geometry only: 22.4 px); grasper hinge 31.9 px (geometry) |
> | **Clean leakage-free L-hook test** (video52+55 never in training, 35 human clicks) | DLC **6.5 px median, 35/35 found, 91 % within 20 px** vs geometry 28.6 px, 25/35 ([docs/DLC_CLEAN_EVALUATION.md](DLC_CLEAN_EVALUATION.md)) |
> | **Bleeding detector** (SurgBlood, split by clip) | detection F1 0.937, AUROC 0.942; in the pipeline and UI |
> | Tissue model for other surgeries (DSAD fine-tune) | DSAD test Dice 0.38 → 0.85, appendicectomy false "liver" ~halved; gallbladder test 0.735 → 0.697, so it is an opt-in **Surgery type: Other**, not the default |
> | **CVS safety model** (Endoscapes surgeon labels, Colab T4) | **test mAP 0.539** on 40 unseen videos ("safe to clip?" estimate, not injury risk) |
> | Risk score | rule-based research score with explanations; learned injury-risk model needs expert event labels |
> | CPU speed | ~3.2 fps default, ~7 fps fast preset (not real-time); DLC runs once per new video |
> | EndoVis robotic DLC (separate experiment) | official test median 3.2 px, recall 57 % (79–92 % on sequences 1–4) |
> | Tests | 117 Python tests + 20 browser (Playwright) tests incl. WCAG 2 AA scan, all passing |
>
> **Web app:** React UI + FastAPI on **http://localhost:8510** — see §9.
>
> DLC v3 was trained on 86 HUMAN L-hook clicks (the CholecSeg8k masks often omit the thin hook, so automatic labels
> stopped short). Details: [docs/DLC_ANNOTATION.md](DLC_ANNOTATION.md) · full report:
> [docs/PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md) · status board: `status/STATUS_BOARD.html` ·
> everything done: [docs/ALL_WORK_DONE.md](ALL_WORK_DONE.md) · EndoVis: [docs/ENDOVIS_INTEGRATION.md](ENDOVIS_INTEGRATION.md)

PRISM-Endo processes laparoscopic/endoscopic video and produces, per frame:
instrument masks (grasper, L-hook), instrument-tip keypoints (DeepLabCut hybrid), tip trajectory and motion,
tissue masks (CholecSeg8k classes), tip-to-tissue proximity, an interpretable risk indicator, and a
Critical View of Safety estimate, written to an annotated video and a CSV.

---

## 1. Architecture

```
input video / CholecSeg8k frame folder      (src/utils/video_io.py — streamed, timestamps kept)
  ├─ A  instrument segmentation            (src/segmentation/, configs/instrument_seg.yaml)
  ├─ B  keypoints                          (src/tracking/)
  │     ├─ DeepLabCut predictions  [primary; needs a trained DLC model]
  │     └─ mask-tip heuristic       [fallback; geometric estimate, clearly labelled]
  ├─ C  motion & trajectory                (src/motion/kinematics.py)
  ├─ D  tissue segmentation                (src/tissue/, configs/tissue_seg.yaml)   [optional]
  ├─ E  tip → tissue-boundary proximity    (src/proximity/)                         [needs D]
  ├─ F  risk: rule-based research score    (src/risk/rule_based.py)
  │          learned model (needs labels)  (src/risk/learned.py)
  └─ G  overlay video + results.csv + summary.json  (src/pipeline/)  · Streamlit UI (src/ui/app.py)
```

Segmentation (pixel masks) and keypoint tracking (points) are separate modules. The
pipeline asks a *keypoint source* for each configured point (`keypoints.points` in
`configs/pipeline.yaml`); a DLC source and a mask-heuristic source implement the same
interface (`src/tracking/sources.py`).

| Folder | Content |
|---|---|
| `src/data` | CholecSeg8k / indexed-PNG discovery, label mapping, clip-wise splits, augmentation |
| `src/segmentation` | model factory (smp U-Net/ResNet-18), training loop, metrics, predictor |
| `src/tracking` | keypoint types, mask-tip heuristic, DLC file reader, DLC workflow wrapper |
| `src/motion` | displacement, velocity, direction, acceleration, turn angle, smoothing, gaps |
| `src/tissue` | tissue model wrapper (reports "unavailable" when no checkpoint) |
| `src/proximity` | inside/outside, boundary distance, closest point, approach speed |
| `src/risk` | rule-based score + learned-model trainer/scorer |
| `src/evaluation` | keypoint error/PCK, warning lead-time / false-alarm metrics |
| `src/pipeline` | orchestrator and overlay drawing |
| `src/ui` | Streamlit app |
| `configs/` | YAML configs (any key can be overridden on the command line: `key.sub=value`) |
| `scripts/` | data/video helpers, DLC workflow, risk-model training, legacy-model check |
| `tests/` | unit + smoke tests (pytest) |
| `outputs/` | checkpoints, runs, logs (generated) |

Legacy material kept unchanged: `files_dataset/` (original reference scaffold; `Synthethgic Videa_dataset/`
is a byte-identical copy), `Dataset_Utilities/`, the notebooks, and `models/unet_model.pth` (see §12).

---

## 2. Environment setup

Tested on Windows 11, Python 3.11.9, 16-core CPU, 16 GB RAM, AMD Radeon RX 6800M (12 GB).
Python 3.14 (the system default here) has no PyTorch build in use; use **Python 3.11**.

Two virtual environments are used because DirectML and DeepLabCut pin different torch versions.
Both are created with `--system-site-packages` so they reuse an existing torch/OpenCV install.

```powershell
# A) .venv  — CPU / CUDA PyTorch + DeepLabCut + Streamlit
py -3.11 -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-dlc.txt        # DeepLabCut 3.0.2 (optional)

# B) .venv-dml — AMD/Intel GPU on Windows via DirectML (used for training here)
py -3.11 -m venv --system-site-packages .venv-dml
.venv-dml\Scripts\python -m pip install -r requirements-directml.txt   # installs torch 2.4.1 + DirectML
.venv-dml\Scripts\python -m pip install streamlit
```

On a machine with an **NVIDIA GPU**, install the CUDA build of torch into `.venv`
(https://pytorch.org/get-started/locally/); `device: auto` then picks CUDA and `.venv-dml` is not needed.

`device: auto` chooses CUDA → DirectML → CPU. Force one with `device=cpu|cuda|dml`.

The first training run downloads ImageNet ResNet-18 encoder weights (~45 MB, via
`segmentation_models_pytorch` from Hugging Face). No datasets are downloaded.

---

## 3. Dataset format and annotation status

`video01/` is **CholecSeg8k** (pixel-annotated frames from Cholec80 video 01):
16 clips × 80 consecutive frames (25 fps) = 1,280 frames of 854×480.

| File | Meaning | Used? |
|---|---|---|
| `frame_N_endo.png` | RGB frame | yes |
| `frame_N_endo_watershed_mask.png` | **dense** class-id map (31 grasper, 32 L-hook, 21 liver, 22 gallbladder, 25 cystic duct, 23 connective tissue, 24 blood, 12 fat, 11 abdominal wall, 13 GI tract, 50 black border; 255 = boundary lines → ignored) | **yes — training labels** |
| `frame_N_endo_color_mask.png` | same, colour-coded | no |
| `frame_N_endo_mask.png` | **sparse** annotation-tool mask (most pixels 0) | **no** — see §12 |

What exists / does not exist:

| Annotation | Status |
|---|---|
| Instrument pixel masks (grasper, L-hook) | ✅ present (CholecSeg8k) |
| Tissue pixel masks (liver, gallbladder, cystic duct, fat, …) | ✅ present; hepatic vein and liver ligament **absent from video01** |
| Instrument-tip **keypoint** labels | ❌ none — must be labelled in DeepLabCut (§5) |
| Risk / near-miss / injury **event labels** | ❌ none — learned risk model cannot be trained (§7) |
| Camera calibration / depth | ❌ none — no physical (mm) distances |
| Multiple patients | ❌ only one patient video → no patient-level generalisation can be measured |

**Splits** (`configs/seg_base.yaml`) are by temporal clip block, never random frames:
test = clips 14859–15099 (4 clips), val = clip 00400, train = the other 11 clips.
Note: cystic duct occurs only in clips 16345–16585 (train), so it cannot be evaluated on the test split.

Other data can be used with `data.layout: indexed_png` (`root/images/<video>_<x>.png` +
`root/masks/<same>.png` storing raw label values); the text before the first `_` is the split group.

---

## 4. Instrument segmentation (Stage A)

Model: U-Net with ImageNet ResNet-18 encoder (`segmentation_models_pytorch`), 3 classes
(background / grasper / l_hook), input 256×448, loss = cross-entropy + soft Dice, AdamW, cosine LR.

```powershell
# smoke test (2 batches) — checks the code path only
.venv\Scripts\python train_segmentation.py --config configs/instrument_seg.yaml --smoke

# train on the AMD GPU (what was run here; batch 8 needs >= ~6 GB free Windows commit memory, see §11)
.venv-dml\Scripts\python train_segmentation.py --config configs/instrument_seg.yaml --max-restarts 5 device=dml train.epochs=15 train.batch_size=8
# low-memory variant (what worked before the page file was enlarged)
.venv-dml\Scripts\python train_segmentation.py --config configs/instrument_seg.yaml --max-restarts 10 device=dml train.epochs=15 train.batch_size=2 train.grad_accum_steps=2

# train on CPU (~5 min/epoch on a 16-core laptop CPU) or CUDA
.venv\Scripts\python train_segmentation.py --config configs/instrument_seg.yaml train.epochs=15

# resume
.venv-dml\Scripts\python train_segmentation.py --config configs/instrument_seg.yaml --resume outputs/models/instrument_seg/last.pt device=dml

# evaluate on the held-out clips (Dice/IoU per class, empty-frame handling, latency)
.venv-dml\Scripts\python evaluate_segmentation.py --config configs/instrument_seg.yaml --checkpoint outputs/models/instrument_seg/best.pt --split test

# predict masks/overlays for a clip, folder, video or image
.venv-dml\Scripts\python predict_segmentation.py --checkpoint outputs/models/instrument_seg/best.pt --input video01/video01_14859 --out outputs/predictions/instrument_14859 --video
```

Checkpoints: `outputs/models/instrument_seg/{best.pt,last.pt,history.csv,train_summary.json}`.
`--max-restarts N` runs training in a child process and resumes from `last.pt` after a crash
(used here because the machine repeatedly ran out of Windows commit memory — §11).

---

## 5. DeepLabCut keypoints (Stage B)

DeepLabCut 3.0.2 (PyTorch engine) is installed in `.venv` and its API is wrapped in
`src/tracking/dlc_workflow.py`. **No keypoints have been labelled yet, so no real DLC model exists.**
Until one does, the pipeline uses the *mask-tip heuristic* and says so in the overlay, CSV
(`*_kp_source = mask_heuristic`) and warnings.

Keypoints are configured in `configs/pipeline.yaml → keypoints.points`
(default: `lhook_tip` on class `l_hook`, `grasper_tip` on class `grasper`). Add more points
(e.g. jaw tips) there; the DLC project takes its bodyparts from the same list.

```powershell
# 0) DLC needs videos: build one per CholecSeg8k clip (H.264 via bundled ffmpeg)
.venv\Scripts\python scripts/make_video_from_frames.py --all video01 --out-dir Videos

# 1) create a project (bodyparts = configured keypoints); use TRAIN-split clips only
.venv\Scripts\python scripts/dlc_workflow.py create --name prism-tips --experimenter me --videos Videos/video01_00240.mp4 Videos/video01_16345.mp4 Videos/video01_16585.mp4 Videos/video01_28660.mp4 Videos/video01_28820.mp4

# 2) select diverse frames (k-means)
.venv\Scripts\python scripts/dlc_workflow.py extract --config outputs/dlc_projects/prism-tips-me-<date>/config.yaml

# 3) label by hand — needs the GUI:  .venv\Scripts\python -m pip install "deeplabcut[gui]==3.0.2"
.venv\Scripts\python scripts/dlc_workflow.py label --config <project>/config.yaml

# 4–6) build training set, train (ImageNet-pretrained ResNet-50 by default), evaluate
.venv\Scripts\python scripts/dlc_workflow.py build --config <project>/config.yaml --net resnet_50
.venv\Scripts\python scripts/dlc_workflow.py train --config <project>/config.yaml --epochs 200 --batch-size 8
.venv\Scripts\python scripts/dlc_workflow.py evaluate --config <project>/config.yaml

# 7) predict on a test video → <video>DLC_*.h5/.csv (+ labelled video)
.venv\Scripts\python scripts/dlc_workflow.py analyze --config <project>/config.yaml --video Videos/video01_14859.mp4 --out outputs/dlc/video01_14859

# 8) use it in the pipeline
.venv-dml\Scripts\python run_pipeline.py --input Videos/video01_14859.mp4 keypoints.source=dlc keypoints.dlc_predictions=outputs/dlc/video01_14859/<file>.h5
```

Labelling rules: label the **visible** tip only; leave a point unlabelled when the tip is
occluded or off-screen (unlabelled = NaN = never treated as ground truth). Label frames from
train-split clips only, so the test clips stay unseen. DLC's own train/test split is by frame;
for a leakage-free estimate, also evaluate on a held-out clip with `scripts/evaluate_keypoints.py`.

Low-confidence handling: DLC likelihood < `keypoints.dlc_min_confidence` (0.6) → status
`low_confidence`; the point is drawn hollow/grey and excluded from motion, proximity and risk.

`scripts/dlc_smoke_test.py` runs the whole DLC chain (create → extract → label → build → train →
analyze → read) on a **synthetic** clip whose tip position is known by construction. It
proves the integration works in this environment; it says nothing about surgical accuracy.

---

## 6. Tissue segmentation (Stage D)

Same training code, 12 classes (CholecSeg8k tissues + one merged `instrument` class so the
model learns occlusion). Classes come from the checkpoint, never hard-coded.

```powershell
.venv-dml\Scripts\python train_segmentation.py --config configs/tissue_seg.yaml --max-restarts 5 device=dml train.epochs=15 train.batch_size=8
.venv-dml\Scripts\python evaluate_segmentation.py --config configs/tissue_seg.yaml --checkpoint outputs/models/tissue_seg/best.pt --split test
```

Without a tissue checkpoint the pipeline runs in **instrument-only mode**: no tissue masks are
drawn, proximity is reported as `unavailable`, and risk is either a clearly labelled
*motion-only* score or disabled (`risk.when_proximity_unavailable: motion_only | disable`).

Proximity targets (`proximity.target_classes`, default `liver, cystic_duct`) are illustrative;
which structures matter is a clinical decision. **Gallbladder is deliberately not a default target**:
in cholecystectomy it is the organ being removed and the grasper holds it on purpose — with it
included, every frame of test clip 14859 scored "Medium" because the grasper tip sat inside it.

---

## 7. Risk scoring (Stage F)

### Mode 1 — rule-based research score (implemented, default)

Per keypoint, features normalised to [0, 1], with distances/speeds as fractions of the image diagonal:

| Feature | Definition (defaults in `configs/pipeline.yaml → risk`) |
|---|---|
| `f_prox` | 1 − d / (0.08·diag); 1 if the tip is inside the target |
| `f_approach` | velocity component towards the nearest boundary / (0.3·diag/s), only if positive |
| `f_speed` | speed / (0.6·diag/s) |
| `f_abrupt` | max(acceleration / (12·diag/s²), turn angle / 180°) |

`score = 0.5·f_prox + g·(0.2·f_approach + 0.15·f_speed + 0.15·f_abrupt)`,
with `g = clip(1 − d / (2·0.08·diag))` so motion only counts near tissue.
Then: exponential smoothing (α = 0.4) → categories Minimal / Low / Medium / High
(thresholds 0.25 / 0.5 / 0.75) with a 0.2 s dwell time and immediate escalation if the
score rises faster than 2.0/s. The frame score is the highest-scoring keypoint.
Every frame stores the feature contributions, an explanation string, and missing-data
status (`no_keypoint`, `low_confidence`, `target_tissue_not_visible`, `tissue_model`).

All constants are plausible defaults, **not calibrated** against any outcome data.

### Mode 2 — learned risk model (implemented, cannot be trained yet)

`scripts/train_risk_model.py` trains a class-balanced logistic regression on causal
windowed features computed from pipeline CSVs (same code online and offline), with
**video-level** splits, and reports precision, recall, F1, AUROC, AUPRC and prevalence.
It refuses to run without a label file, with fewer than 2 videos, or with too few
positive events. Label format (CSV):

```
video_id,start_s,end_s,label        # interval form: frames inside an interval get `label`, the rest 0
video_id,source_index,label         # frame form
```

`video_id` = the input file/folder name used by the pipeline. Labels must come from expert
review (e.g. surgeon-marked near-miss / unsafe-approach events) with a written definition.
Event-level warning metrics (lead time, false alarms/min) are in `src/evaluation/warnings.py`.

---

## 8. Running the pipeline

```powershell
# demo videos from the CholecSeg8k clips
.venv\Scripts\python scripts/make_video_from_frames.py --all video01 --out-dir Videos

# full pipeline on a held-out test clip (GPU)
.venv-dml\Scripts\python run_pipeline.py --input Videos/video01_14859.mp4

# directly on a frame folder (timestamps from frame numbers at 25 fps)
.venv-dml\Scripts\python run_pipeline.py --input video01/video01_14859

# options
... run_pipeline.py --input X --max-frames 100 tissue.checkpoint=null           # instrument-only
... run_pipeline.py --input X risk.when_proximity_unavailable=disable
... run_pipeline.py --input X keypoints.source=dlc keypoints.dlc_predictions=path.h5
```

Outputs in `outputs/runs/<name>/`: `annotated.mp4`, `annotated_h264.mp4` (browser-playable),
`original_h264.mp4`, `results.csv` (one row per frame: keypoints + status/source, smoothed
position, displacement/velocity/direction/acceleration/turn angle, per-target distances,
inside flags, approach speed, risk score/category/explanation/contributions, per-stage ms),
`summary.json` (modes, warnings, runtime, hardware).

## 9. Interface

### Web app (React) — main interface

```powershell
cd web; npm install; npm run build; cd ..                                        # once, or after UI changes
.venv-dml\Scripts\python -m uvicorn src.api.server:app --port 8510 --host 127.0.0.1  # API + built UI
# open http://localhost:8510
cd web; npm run test:e2e                                                        # 20 browser tests (server must run)
```

- **Sidebar:** upload (drag and drop), video, surgery type (tissue model), tip source (Hybrid default /
  DLC only / Geometry only), max frames, **tissue annotation** (distance targets with tested / not-tested markers,
  outline all tissues, overlay strength), Analyse with live progress and **Stop**, saved analyses.
- **Results:** KPI cards, synced annotated + original video, "Why this risk", stacked timeline (risk, bleeding, CVS,
  grasper, L-hook, near target; drag-zoom, milestones, reviewer flags, thumbnails), summary report (print / PDF),
  events, reviewer flags (saved to `review_flags.json` next to the run), signal charts (risk chart coloured by
  risk level), frame inspector, analysis settings, compare two analyses, downloads, light / dark theme.
- **Keyboard:** Space play/pause · D/F frame (Shift: 1 s) · J/K previous/next event · 1/2/3 flags · T theme · ? help.
- Runs write to a temporary folder and replace results only when finished; Stop discards the run. Models are
  cached between runs; first DLC analysis of a new video is cached in `outputs/dlc/auto_analyze/`.
- Code: `web/` (React 18, Vite, Tailwind 4, Geist), `src/api/server.py` (FastAPI). Tests: `web/e2e/app.spec.ts`.
- Sharing: a temporary Cloudflare quick tunnel (`tools/cloudflared.exe tunnel --url http://localhost:8510`) gives a
  public link with **full access**; the link changes when the tunnel restarts. Antigravity's Ports panel cannot
  forward ports (no `code-tunnel.exe`); a fixed URL needs a Microsoft devtunnel / ngrok / Cloudflare account.

### Streamlit app (legacy)

```powershell
.venv-dml\Scripts\streamlit run src/ui/app.py      # GPU (DirectML)
.venv\Scripts\streamlit run src/ui/app.py          # CPU/CUDA
```

Select a video/clip, upload one, or reopen a previous run; set checkpoints, keypoint source,
proximity targets and risk options; run with a progress bar; view original and annotated
video, risk/distance/speed time series, a frame inspector with per-frame explanations, and
download the CSV, summary and video. Missing models are shown as warnings.

## 10. Tests and evaluation commands

```powershell
.venv\Scripts\python -m pytest tests -q                 # unit + smoke tests
.venv\Scripts\python scripts/evaluate_legacy_unet.py --size 256
.venv\Scripts\python scripts/evaluate_keypoints.py --gt <CollectedData.h5> --pred <results.csv|DLC.h5> --bodyparts lhook_tip --width 854 --height 480
.venv\Scripts\python scripts/dlc_smoke_test.py --epochs 1
```

### Measured results (2026-10-10, this machine)

All segmentation numbers are on the **held-out test clips 14859–15099** (320 frames, never used for
training or model selection), at full 854×480 resolution, dataset-level Dice/IoU. One patient only.

**Instrument segmentation** (`outputs/models/instrument_seg/best.pt`, 15 epochs, best val Dice 0.936 on clip 00400)

| Class | Dice | IoU |
|---|---|---|
| grasper | 0.835 | 0.717 |
| L-hook | 0.930 | 0.870 |
| mean | **0.883** | 0.793 |

Frame-level presence (≥200 px): grasper precision/recall 1.00/1.00; L-hook 0.994/1.00 (2 false-positive frames).

**Tissue segmentation** (`outputs/models/tissue_seg/best.pt`, 15 epochs, best val mean Dice 0.917)

| Class | Dice | IoU | Note |
|---|---|---|---|
| liver | 0.934 | 0.875 | |
| abdominal wall | 0.913 | 0.839 | |
| instrument (occluder) | 0.903 | 0.823 | |
| gallbladder | 0.867 | 0.765 | |
| connective tissue | 0.757 | 0.609 | |
| blood | 0.178 | 0.098 | only 62k GT pixels in the whole split |
| mean over the 7 classes present | **0.759** | | ≈0.89 without blood |
| fat, GI tract | — | — | absent from test GT (model predicted a few false-positive pixels) |
| cystic duct, hepatic vein, liver ligament | — | — | absent from test GT → not evaluable |

**Legacy notebook model**: see §12 (it learned "annotated area", not instruments).

**Keypoints**: not measurable — no tip annotations exist. DLC integration verified end-to-end on a
synthetic clip (`scripts/dlc_smoke_test.py`, 2 epochs, ~2 min on CPU): project creation, frame
extraction, training-set creation, training, video analysis and PRISM-Endo's reader all work with
deeplabcut 3.0.2; the pipeline consumed the DLC `.h5` (58 frames `ok`, 2 gated as `low_confidence`).

**Risk**: no labels → precision/recall/AUROC/lead time cannot be computed. On test clip 14859 the
rule-based score was Low in 44 and Medium in 36 frames, driven by the L-hook working 12–28 px from
the liver boundary (example explanation: *"lhook_tip: Medium - tip 12px from liver (+0.43); abrupt
trajectory change (+0.09)"*). This shows the score responds as designed, not that it is correct.

**Runtime** (AMD RX 6800M via DirectML unless noted)

| What | Measured |
|---|---|
| Instrument model inference | 7.5 ms/frame (GPU) · 135 ms/frame (CPU) |
| Tissue model inference | 8.1 ms/frame (GPU) |
| Full pipeline, 854×480, both models + overlay + CSV | 12–15 fps (≈70–80 ms/frame, most of it overlay drawing) |
| Full pipeline, instrument-only on CPU | 6.6 fps |
| Training, 880 frames/epoch at 256×448 | ≈40–45 s/epoch (GPU) · ≈5 min/epoch (CPU, batch 4) |

**Tests**: `pytest tests` → 54 passed (motion, proximity, risk, tracking/DLC IO, segmentation
metrics/splits, keypoint & warning evaluation, learned-risk guards, pipeline smoke tests incl.
instrument-only, risk-disabled, no-model and CSV-schema regression). The Streamlit app was
checked headlessly with `streamlit.testing` (open a run, frame inspector, run the pipeline).

## 11. CPU / GPU guidance

* **NVIDIA (CUDA)**: fastest; batch 8–16 at 256×448 fits in 4–6 GB VRAM.
* **AMD/Intel on Windows (DirectML, `.venv-dml`)**: used here. Notes from this machine:
  * data-loader workers (`num_workers>0`) caused DirectML "allocator out of memory" / device-removed
    errors → forced to 0 on DirectML;
  * torch's AdamW runs `lerp_` on the CPU under DirectML → a DirectML-friendly AdamW
    (`src/segmentation/optim.py`) is used automatically;
  * smp's DiceLoss failed under DirectML autograd → a float-only Dice loss is used;
  * **GPU allocations count against Windows commit memory (RAM + page file).** With the original
    fixed 8 GB page file (~4 GB free commit) batch 8 crashed (~5.5 GB), batch 4 (~3.4 GB) crashed
    intermittently, batch 2 + `grad_accum_steps=2` (~2.1 GB) was stable. After enlarging the page
    file to 16–32 GB (commit limit 23 → 31 GB) batch 8 trained without problems.
    To change it: `sysdm.cpl` → Advanced → Performance Settings → Advanced → Virtual memory →
    Custom size 16384 / 32768 MB (or *System managed*), then reboot.
  * Never run two GPU training processes at once on a 16 GB machine.
* **CPU**: fine for inference on short clips and smoke tests; training is ~5 min/epoch
  (16-core laptop CPU, batch 4). `train.num_threads` caps CPU threads.

## 12. Limitations and troubleshooting

* **Single patient.** All 1,280 frames come from one Cholec80 video; scores measure
  generalisation across time within one patient only. Expect lower performance on new patients.
* **Legacy notebook model** (`models/unet_model.pth`): the notebook built its target as
  `_endo_mask.png > 0`, i.e. "any annotated pixel", not "instrument". Measured with
  `scripts/evaluate_legacy_unet.py` (test clips, every 8th frame): it predicts ~27–34 % of the
  image as foreground (instruments cover ~8 %), Dice vs instrument 0.26–0.32, Dice vs
  "any annotated" 0.50–0.68. It is not used by the pipeline. (Its random frame-level split also
  means its own reported numbers would be optimistic.)
* **Keypoints are heuristic until DLC is trained.** The mask-tip heuristic fails for
  crossing/merged instruments, curved tools, and masks touching the image edge at the tip; it
  jitters frame to frame (visible as a jagged trail), and with two instruments of the same class
  (e.g. two graspers, common in these clips) it only tracks the largest one. No tip ground truth
  exists, so its accuracy has **not** been measured.
* **Empty-frame handling is barely tested on held-out data**: both instruments are visible in
  almost every test frame (only 2 test frames lack an L-hook; both got false-positive L-hook pixels).
  On clip 00080 (mostly no instrument, but a training clip) predicted presence matched ground truth
  exactly (grasper 10/10 frames, L-hook 0/0) and the 70 empty frames produced no keypoints and risk "Unavailable".
* **2-D image space only.** No depth/calibration → no mm distances; camera motion is included in tip motion;
  "distance to tissue" is distance to the nearest *visible* pixel of that class (instruments occlude tissue).
* **Risk score is uncalibrated** and has never been compared with real adverse events.
* **No temporal model** for segmentation (frame-by-frame); masks can flicker.
* Troubleshooting: `Checkpoint not found` → train first or fix the path; video won't play in
  the browser → ensure `imageio-ffmpeg` is installed (H.264 transcode); DLC import errors →
  use `.venv`; DirectML crashes → see §11.

---

## 13. Update 2026-10-10 — CPU continuation (implemented / tested / pending)

Performance numbers in §10 labelled "GPU" were measured on this laptop's AMD RX 6800M via DirectML
(there is no CUDA). Everything in this section is **CPU-only** (AMD Ryzen 9 5980HX, 8 cores / 16 threads, `.venv`).

### Status

| Item | State |
|---|---|
| DLC annotation projects (200 train frames from train clips, 40 eval frames from test clips), guide, label gate, held-out evaluation script | **Implemented and run**. **No labels exist**, so DLC is not trained and tip accuracy is not measured |
| Multi-video data loading (`data.root` list, `data.group_by: video`), `scripts/make_splits.py` | Implemented and tested; with one video the script **refuses** a patient-level split |
| Risk-event schema (`configs/risk_events.yaml`, *proposal*), validator, training gate, annotation UI (`src/ui/annotate_events.py`) | Implemented and tested; no labels, so the **learned model is not trained** |
| Tissue target reliability (`class_support.json`) | Implemented; untrained targets refused; cystic duct flagged "not validated" (`*` on the overlay) |
| Temporal mask smoothing, camera-motion estimate, compensated tip speed, two-grasper tracking | Implemented, tested, measured (below) |
| CPU profiling and fast preset | Measured (below) |
| Tests | `pytest tests`: **82 passed** |

### Commands

```powershell
.venv\Scripts\python scripts\prepare_dlc_annotation.py --total 200 --eval-total 40   # already run; see docs/DLC_ANNOTATION.md
.venv\Scripts\python scripts\dlc_workflow.py status --config outputs\dlc_projects\prism-tips-train-annotator-2026-10-10\config.yaml
.venv\Scripts\python scripts\evaluate_dlc_heldout.py --eval-config outputs\dlc_projects\prism-tips-eval-annotator-2026-10-10\config.yaml --pipeline-runs outputs\runs
.venv\Scripts\python scripts\make_splits.py --out outputs\splits_proposal.yaml                # docs/DATASETS.md
.venv\Scripts\python scripts\validate_risk_labels.py <labels.csv> --runs-dir outputs\runs     # docs/RISK_ANNOTATION.md
.venv\Scripts\python scripts\class_support.py --config configs\tissue_seg.yaml --checkpoint outputs\models\tissue_seg\best.pt
.venv\Scripts\python scripts\evaluate_temporal.py --split test
.venv\Scripts\python run_pipeline.py --input Videos\video01_14859.mp4 --config configs\pipeline_cpu_fast.yaml
```

### Annotation facts that limit what can be learned (measured from the ground-truth masks)

* **Two graspers:** 13 of 880 training frames show two graspers, but all 320 test frames do.
  `grasper_tip_2` is optional in the label gate, and grasper tips are evaluated identity-free.
* **Tissue support:** `hepatic_vein` and `liver_ligament` have 0 training frames (cannot be detected; refused as
  targets). `cystic_duct` (239 train frames), `fat` and `gastrointestinal_tract` have **no test frames**, so
  their detection quality is unknown. Distances stay in **pixels**; no calibration or depth exists.

### Flicker reduction (held-out test clips, 320 frames)

"Spurious change" is the share of pixels whose predicted label changes between consecutive frames while the
ground truth does not.

| Variant | Instrument Dice | Instr. spurious | Tissue Dice | Tissue spurious |
|---|---|---|---|---|
| no smoothing | 0.882 | 0.61 % | 0.759 | 1.97 % |
| **EMA α=0.7 (new default)** | 0.883 | 0.53 % | 0.759 | 1.56 % |
| EMA α=0.5 | 0.880 | 0.52 % | 0.757 | 1.35 % |
| EMA α=0.7 + camera-motion warp | 0.882 | 0.56 % | 0.760 | 1.65 % |

* **Camera motion** in these clips is small: median 1.3 px/frame, 90th percentile 4.5 px, rotation ~0.07°.
  The LK + RANSAC fit was reliable for 316/316 frame pairs (median residual 0.3 px) and costs ~5 ms.
  Warping the smoothing history did **not** help, so it is off. Camera-compensated tip speed is written as
  `*_speed_comp_px_s` but is not used for risk (`risk.use_camera_compensated_speed: false`); it has not
  been validated against ground-truth camera motion (none exists).
* **Two-grasper tracking:** `grasper_tip` and `grasper_tip_2` are matched frame to frame (Hungarian,
  150 px gate). Merged masks of touching instruments cannot be separated.

### CPU performance (test clip 14859, 80 frames, 854×480, full pipeline incl. overlay video)

| Configuration | FPS | ms/frame | Instrument | Tissue | Keypoints | Overlay/CSV |
|---|---|---|---|---|---|---|
| default `pipeline.yaml` | 2.87 | 348 | 134 | 155 | 26 | 33 |
| upsample = nearest | 3.44 | 290 | 117 | 119 | 22 | 33 |
| tissue every 2nd frame | 3.91 | 255 | 128 | 73 | 21 | 32 |
| no overlay video | 3.23 | 308 | 131 | 152 | 20 | 5 |
| **`pipeline_cpu_fast.yaml`** (nearest + tissue/2 + 16 threads) | **4.8–5.0** | ~205 | 98 | 52 | 22 | 35 |

Model forward alone (instrument U-Net, 256×448): 125 ms at 4 threads, 86 ms at 8, 72 ms at 16;
`channels_last` was slower (116 ms) and is not used. Labels-only prediction now skips the full-resolution
softmax (identical output), and tissue boundaries are extracted once per frame. **The pipeline is not
real-time on this CPU** (25 fps needs 40 ms/frame). `nearest` upsampling costs Dice 0.004 (instrument) /
0.001 (tissue); the accuracy effect of `tissue.every_n_frames=2` has not been measured.

### Next required actions

1. Label the 200 + 40 DLC frames (docs/DLC_ANNOTATION.md), then `dlc_workflow.py build/train` and `evaluate_dlc_heldout.py`.
2. Obtain ≥2 more CholecSeg8k/Cholec80 videos and retrain with `data.group_by: video` (docs/DATASETS.md).
3. Have a surgeon confirm `configs/risk_events.yaml`, annotate whole videos blind (≥2 annotators), validate, then train.
4. Validate cystic-duct segmentation on held-out data (needs a clip with cystic duct outside training).
5. Optional: ONNX Runtime CPU inference (onnxruntime 1.21 is installed, not yet tested).

### DLC labelling status (checked 2026-10-10 03:00)

`dlc_workflow.py status` on both projects: **0 of 240 frames labelled** (train 200 frames in 11 folders,
eval 40 frames in 4 folders). DLC has **not** been trained and no keypoint accuracy exists yet.

Labelling tool (separate env `.venv-label`: napari 0.9.2 + napari-deeplabcut 0.4.0, so `.venv` stays untouched):

```powershell
.venv-label\Scripts\python scripts\label_frames.py --project train   # opens the next unlabelled training folder
.venv-label\Scripts\python scripts\label_frames.py --project eval    # then the 4 held-out folders
.venv\Scripts\python scripts\dlc_workflow.py status --config outputs\dlc_projects\prism-tips-train-annotator-2026-10-10\config.yaml
```

Once `status` reports at least 100 labelled training frames (at least 30 for `lhook_tip` and `grasper_tip`):
`dlc_workflow.py build` → `train --device cpu` → `analyze` on the test videos → `scripts/evaluate_dlc_heldout.py`.
