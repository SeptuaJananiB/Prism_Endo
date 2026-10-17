# PRISM-Endo · Download links

Everything that is **not** inside the handover zip (`final_endo.zip`), where to get it, and where to put it.

> **You do not need any of this to run the app.** The zip already has the code, all trained models, the test videos
> and `START.bat`. These downloads are only for **re-training, re-evaluating or extending** the models.

---

## 1. Project code

| What | Link |
|---|---|
| Source code, docs, issues (always the newest version) | <https://github.com/SeptuaJananiB/Prism_Endo> |

---

## 1b. Trained models and test videos

| What | Link |
|---|---|
| All trained models (~1 GB, unzip into the project folder; `START.bat` does this automatically) | <https://github.com/SeptuaJananiB/Prism_Endo/releases/download/models-v1/prism-endo-models.zip> |
| 23 test videos (~16 MB, short CholecSeg8k clips incl. the held-out tip-evaluation clips; unzip into the project folder, `START.bat` does this automatically) | <https://github.com/SeptuaJananiB/Prism_Endo/releases/download/models-v1/prism-endo-test-videos.zip> |
| Release page with the model list and results | <https://github.com/SeptuaJananiB/Prism_Endo/releases/tag/models-v1> |

So the project can be rebuilt from GitHub alone: download the code, double-click `START.bat` (it fetches the models).

---

## 2. Training and evaluation datasets

Put each dataset in the folder shown (inside the project folder). Sizes are what this project used.

| Dataset | Used for | Put it in | Size | Download | Access | Licence |
|---|---|---|---|---|---|---|
| **CholecSeg8k** | instrument + tissue segmentation (main training data) | `data/cholecseg8k/` (first video also as `video01/` in the project root) | ~6 GB | <https://www.kaggle.com/datasets/newslab/cholecseg8k> | free Kaggle account | CC BY-NC-SA 4.0 |
| **Endoscapes (CVS201)** | Critical View of Safety model, cystic-duct check | `data/endoscapes/` | ~25 GB | <https://github.com/CAMMA-public/Endoscapes> (download links in its README) | free | CC BY-NC-SA 4.0 |
| **DSAD** (Dresden Surgical Anatomy Dataset) | tissue model for other surgeries ("Other abdominal") | `data/dsad/` | ~2.3 GB | <https://huggingface.co/datasets/nct-tso/dsad> | free Hugging Face account | CC BY 4.0 |
| **SurgBlood** | bleeding detector | `data/surgblood/` | ~250 MB | <https://github.com/PJLallen/SurgBlood> | see the repository | see dataset terms |
| **EndoVis 2015 instrument tracking** (Du et al.) | robotic keypoint experiment | `data/endovis15_test/` | ~75 MB | <https://endovissub-instrument.grand-challenge.org/> | free grand-challenge account | see challenge terms |
| **Cholec80** *(optional, more videos)* | extra cholecystectomy videos | `data/cholec80/` | ~70 GB | <https://camma.unistra.fr/datasets/> | request form | CC BY-NC-SA 4.0 |
| **PhaKIR** *(next step)* | instrument keypoints (better grasper tips), phases, instrument masks | `data/phakir/` | large | <https://zenodo.org/records/15740620> · frame-extraction script: <https://github.com/remic-othr/PhaKIR_Dataset> | Zenodo account + access request | see dataset terms |

Data layout and splits: [`docs/DATASETS.md`](docs/DATASETS.md). Training commands: [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md).

> Models trained on **NC** (non-commercial) datasets inherit that restriction: research and teaching use only.

---

## 3. Software

| What | Needed for | Link |
|---|---|---|
| **Python 3.11** (tick *Add python.exe to PATH*) | running the app (required) | <https://www.python.org/downloads/release/python-3119/> |
| PyTorch | installed automatically by `START.bat`; GPU builds (NVIDIA) | <https://pytorch.org/get-started/locally/> |
| PyTorch DirectML | AMD / Intel GPU on Windows (`requirements-directml.txt`) | <https://learn.microsoft.com/windows/ai/directml/pytorch-windows> |
| **DeepLabCut** | the DeepLabCut tip models (optional; without it the app uses its built-in tip method) | `pip install "deeplabcut[pytorch]"` · <https://github.com/DeepLabCut/DeepLabCut> |
| Node.js 20+ | only to change and rebuild the web app (`cd web; npm install; npm run build`) | <https://nodejs.org/> |
| Groq API key | the AI explanation (a key is already in `.env`; get your own if it stops working) | <https://console.groq.com/keys> |

---

## 4. Things with no download link

These were created on the original computer during training. They are **not needed** to run or use the project; the
final trained models are already in the zip.

| Left out | What it is | If you need it |
|---|---|---|
| `outputs/dlc_projects/` (19 GB) | DeepLabCut training projects (labelled frames, intermediate files) | Re-create with the DLC scripts in `scripts/` after downloading the datasets (see `docs/TECHNICAL_GUIDE.md`) |
| `last.pt` files | "resume training" checkpoints | Only to continue an interrupted training run; re-train instead |
| `outputs/models/dlc_v2_keep/` | old, rejected DeepLabCut model backup | Not needed (not in the models download) |
| `.venv*` folders | Python environments | `START.bat` re-creates them |
| `outputs/runs/` | saved analyses from the original computer | Re-analyse any video in the app |

> The trained models are also published on GitHub (section 1b), so they stay downloadable even without the zip.
