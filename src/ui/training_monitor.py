"""PRISM-Endo live dashboard: system + watchdog, running jobs with ETA, segmentation runs and all measured results.

  .venv\\Scripts\\streamlit run src/ui/training_monitor.py --server.port 8600

Read-only: it only reads logs, CSV/JSON outputs and system counters.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
MODELS, LOGS, DLC = OUT / "models", OUT / "logs", OUT / "dlc_projects"

st.set_page_config(page_title="PRISM-Endo dashboard", layout="wide")
st.title("PRISM-Endo dashboard")
refresh = st.sidebar.slider("Auto-refresh (s)", 10, 120, 30)
st.caption(f"Updated {datetime.now():%H:%M:%S} · refreshes every {refresh}s")


def fmt(sec):
    if sec is None or sec != sec:
        return "estimating..."
    h, m = divmod(int(sec // 60), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def read_json(p):
    try:
        t = Path(p).read_text(encoding="utf-8", errors="replace")
        return json.loads(t[t.index("{"):t.rindex("}") + 1])
    except Exception:
        return None


# ---------------------------------------------------------------- system + watchdog
st.header("System")
try:
    import psutil

    vm, sw, bat = psutil.virtual_memory(), psutil.swap_memory(), psutil.sensors_battery()
    c = st.columns(4)
    c[0].metric("CPU", f"{psutil.cpu_percent(interval=0.5):.0f} %")
    c[1].metric("Free RAM", f"{vm.available / 2**30:.1f} GB")
    c[2].metric("Free commit", f"{(vm.available + sw.free) / 2**30:.1f} GB")
    c[3].metric("Battery", f"{bat.percent}% {'plugged' if bat.power_plugged else 'ON BATTERY'}" if bat else "n/a")
except Exception as e:
    st.info(f"System counters unavailable: {e}")
wd = LOGS / "watchdog.log"
if wd.is_file():
    lines = wd.read_text(encoding="utf-8", errors="replace").splitlines()
    alive = time.time() - wd.stat().st_mtime < 400
    st.write(f"**Watchdog:** {'running' if alive else 'NOT RUNNING (no log update for >6 min)'}"
             f" · throttled now: {'yes' if lines and 'throttled=True' in ' '.join(lines[-3:]) else 'no'}")
    with st.expander("Watchdog events (latest 12)"):
        st.code("\n".join(lines[-12:]))

# ---------------------------------------------------------------- DLC jobs
st.header("DeepLabCut jobs")
overall_left = 0.0
overall_slot = st.empty()
MEASURED_EPOCH_S = {"endovis-robotic-*": 280, "prism-tips-PSEUDO-*": 90, "prism-tips-AUTO-*": 90, "prism-tips-CLEANs1-*": 250, "prism-tips-CLEANs2-*": 210, "prism-tips-V3fold-*": 220, "prism-tips-V3humanlhook-*": 210, "prism-tips-MULTI16v2-*": 420}  # measured GPU epoch times
resume_from = {"endovis-robotic-*": 41, "prism-tips-PSEUDO-*": 16}  # first epoch of the resumed runs
JOBS = [
    ("CLEAN TEST stage 1: DLC from ImageNet, video52/55 fully excluded (GPU)", "prism-tips-CLEANs1-*",
     LOGS / "clean_eval.log", 45, None),
    ("CLEAN TEST stage 2: fine-tune on your L-hook clicks (4 videos), then frozen test on video52/55 (GPU)",
     "prism-tips-CLEANs2-*", LOGS / "clean_eval.log", 70, OUT / "fold_eval_clean" / "fold_results.json"),
    ("WIDER TEST: DLC without video52/55, tested on your clicks from those 2 unseen patients (GPU)",
     "prism-tips-V3fold-*", LOGS / "v3fold_dlc.log", 115, OUT / "fold_eval" / "fold_results.json"),
    ("DLC v3: trained on YOUR 86 L-hook clicks (GPU) - used in the app hybrid", "prism-tips-V3humanlhook-*",
     LOGS / "v3_dlc.log", 115, LOGS / "v3_dlc_eval_c0.0.log"),
    ("DLC v2 + settling: 1,387 automatic labels, 16 videos (GPU)", "prism-tips-MULTI16v2-*",
     LOGS / "multi16v2_dlc.log", 80, LOGS / "multi16v2_dlc_eval_c0.0.log"),
    ("EndoVis robotic (GPU, separate experiment)", "endovis-robotic-*", LOGS / "endovis_gpu_resume.log", 79,
     OUT / "endovis" / "eval_long" / "endovis_heldout_metrics.json"),
    ("Laparoscopic tips, trained on MACHINE labels (GPU, prototype)", "prism-tips-PSEUDO-*", LOGS / "pseudo_dlc_resume.log",
     30, LOGS / "pseudo_dlc_eval.log"),
    ("Laparoscopic tips, trained on AUTO pseudo-labels (auto_geo_v1), scored on 40 HUMAN labels (GPU)", "prism-tips-AUTO-*",
     LOGS / "auto_dlc.log", 60, LOGS / "auto_dlc_eval_c0.5.log"),
]
# Tissue fine-tune with DSAD (fix false "liver" outside cholecystectomy) - FINISHED, results only
st.subheader("TISSUE: CholecSeg8k + DSAD fine-tune - fix false liver on other surgeries (finished)")
_rows = []
for _name, _dir, _where in (("Colab T4 run", "tissue_seg_dsad_colab", "Colab T4, 1/4 of CholecSeg8k"),
                            ("Guarded local run", "tissue_seg_dsad_local", "laptop GPU, full data, gallbladder guard")):
    _r = read_json(OUT / "models" / _dir / "test_results.json")
    if _r:
        _o, _f = _r.get("original", {}), _r.get("dsad_finetuned", {})
        _rows.append({"Run": _name, "Trained on": _where,
                      "Gallbladder test Dice (original -> new)": f"{_o.get('cholec_test_mean_dice', 0):.3f} -> {_f.get('cholec_test_mean_dice', 0):.3f}",
                      "Other-surgery (DSAD) test Dice": f"{_o.get('dsad_test_mean_dice', 0):.3f} -> {_f.get('dsad_test_mean_dice', 0):.3f}"})
if _rows:
    st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")
st.success("Decision: not adopted as the default (gallbladder surgery got ~0.04 Dice worse). The Colab model is "
           "available as Surgery type = Other (appendix, bowel): false 'liver' on the appendicectomy video ~halved "
           "(9.9 % -> 4.6 % of pixels). The original model stays the gallbladder default.")
st.caption("The run in outputs/models/tissue_seg_dsad stopped at epoch 2 when the laptop switched off; it was "
           "replaced by the two finished runs above.")

for title, pattern, log, total, result in JOBS:
    projs = sorted(DLC.glob(pattern))
    stats = sorted(projs[-1].glob("dlc-models-pytorch/*/*/train/learning_stats.csv")) if projs else []
    st.subheader(title)
    if not stats:
        st.info("not started")
        continue
    df = pd.read_csv(stats[0])
    done = int(df["step"].iloc[-1])  # last row: resumed runs can append after rows of an interrupted run
    log_txt = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    m = re.search(r"start (?:\d{4}-\d\d-\d\d )?(\d\d:\d\d:\d\d)", log_txt)
    finished = "train exit" in log_txt
    queued = not log.is_file()
    eta = None
    if queued:
        eta = (total - done) * MEASURED_EPOCH_S.get(pattern, 300)
    elif m and not finished:
        start = datetime.combine(datetime.now().date(), datetime.strptime(m[1], "%H:%M:%S").time()).timestamp()
        first = resume_from.get(pattern, int(df["step"].min()))
        n_run = max(1, done - first + 1)
        per = (stats[0].stat().st_mtime - start) / n_run
        eta = (total - done) * per if per > 0 else None
    c = st.columns(4)
    c[0].metric("Epoch", f"{done} / {total}")
    loss_col = "losses/train.total_loss"
    if loss_col in df:
        c[1].metric("Train loss", f"{df[loss_col].dropna().iloc[-1]:.5f}")
    rmse = [k for k in df.columns if "rmse" in k.lower() and "pcutoff" not in k.lower()]
    if rmse and df[rmse[0]].notna().any():
        c[2].metric("Internal RMSE (px)", f"{df[rmse[0]].dropna().iloc[-1]:.2f}",
                    help="DLC's own split of the TRAINING sequences - not the held-out score")
    if eta is None and not finished:
        eta = (total - done) * MEASURED_EPOCH_S.get(pattern, 300)
    c[3].metric("Time left", "finished" if finished else (f"queued · ~{fmt(eta)}" if queued else fmt(eta)))
    if not finished:
        overall_left += eta or 0
    st.progress(min(1.0, done / total))
    if loss_col in df:
        st.line_chart(df.set_index("step")[[loss_col]].dropna())
    res = read_json(result) if result else None
    if res:
        st.success("Held-out result available")
        st.json(res, expanded=False)

overall_slot.metric("Estimated time until all jobs finish", "all finished" if overall_left == 0 else fmt(overall_left),
                    help="Training only; each evaluation adds ~5-15 min")

# ---------------------------------------------------------------- cloud job
st.header("Cloud GPU job (Google Colab, Tesla T4)")
st.info("CVS safety model (Endoscapes, surgeon labels): runs in your Colab notebook PRISM_Endo_CVS_colab.ipynb "
        "(Google Drive). Progress is shown there; results are copied here when it finishes "
        "(outputs/models/cvs_resnet50/cvs_report.json).")
cvs = read_json(MODELS / "cvs_resnet50" / "cvs_report.json")
if cvs:
    final = cvs.get(cvs.get("final", ""), cvs)  # report holds v1 and v2; "final" names the chosen one
    test_map = (final.get("test") or {}).get("mAP")
    if test_map is not None:
        st.success(f"CVS model ({cvs.get('final', 'v1')}), unseen test videos: mAP {test_map:.3f} - finished")

# ---------------------------------------------------------------- TipNet (heatmap keypoint network)
st.header("TipNet keypoint network (GPU, trained on automatic pseudo-labels of 16 videos)")
tlog, thist = LOGS / "tipnet_train.log", MODELS / "tipnet" / "history.csv"
if thist.is_file():
    h = pd.read_csv(thist)
    total = 15
    c = st.columns(4)
    c[0].metric("Epoch", f"{int(h.epoch.iloc[-1])} / {total}")
    c[1].metric("Val L-hook median", f"{h.lhook_tip_median_px.iloc[-1]:.1f} px" if h.lhook_tip_median_px.notna().iloc[-1] else "-")
    c[2].metric("Val grasper median", f"{h.grasper_hinge_median_px.iloc[-1]:.1f} px"
                if h.grasper_hinge_median_px.notna().iloc[-1] else "-")
    finished = tlog.is_file() and "done;" in tlog.read_text(errors="replace")
    c[3].metric("Time left", "finished" if finished else fmt((total - len(h)) * h.sec.mean()))
    st.progress(min(1.0, len(h) / total))
    st.caption("Validation = videos 20/26/43 vs automatic labels (not human). Human score: models/tipnet/human_eval.json")
    st.dataframe(h.round(3), width="stretch")
    st.line_chart(h.set_index("epoch")[["lhook_tip_median_px", "grasper_hinge_median_px"]])
    he = read_json(MODELS / "tipnet" / "human_eval.json")
    if he:
        st.success(f"On 40 HUMAN frames: L-hook median {he['lhook']['median_px']:.1f} px "
                   f"(recall {he['lhook']['detection_recall']:.0%}), grasper median {he['grasper']['median_px']:.1f} px "
                   f"(recall {he['grasper']['detection_recall']:.0%})")
elif tlog.is_file():
    st.info("Epoch 1 running (about 6 min per epoch)...")
st.markdown("**Current best on 40 human frames (live pipeline):** L-hook 22.4 px · grasper hinge 31.9 px "
            "(was 128 px before this round)")

# ---------------------------------------------------------------- measured results
st.header("Measured results (held-out data only)")
r = st.columns(2)
ev = read_json(OUT / "endovis" / "eval" / "endovis_heldout_metrics.json")
if ev:
    o = ev["overall"]
    pb = {k: v for k, v in ev["per_bodypart"].items() if v["labelled"]}
    rec = sum(v["detection_recall"] * v["labelled"] for v in pb.values()) / max(1, sum(v["labelled"] for v in pb.values()))
    r[0].markdown(f"**EndoVis DLC, 19 epochs (Dataset4)**  \nrecall {rec:.0%} · median {o['median_px']:.1f} px · "
                  f"PCK@10 {o['pck@10px']:.2f} (on detected joints)")
tb = read_json(OUT / "label_qa" / "tip_baseline_metrics.json")
if tb:
    r[1].markdown(f"**Mask-tip method vs 40 human-labelled frames**  \nL-hook median {tb['lhook']['median_px']:.1f} px "
                  f"(recall {tb['lhook']['detection_recall']:.0%}) · grasper {tb['grasper']['median_px']:.0f} px "
                  "(definition mismatch: jaw hinge vs jaw ends)")
rows = []
for name in ("instrument_seg_multivideo", "tissue_seg_multivideo", "instrument_seg", "tissue_seg"):
    for sub in ("eval_test", "eval_unseen_videos"):
        mj = read_json(MODELS / name / sub / "metrics.json")
        if mj:
            rows.append({"model": name, "eval": sub, "split": ", ".join(mj.get("groups", [])),
                         "mean Dice (present fg)": round(mj["mean_dice_present_fg"] or 0, 3),
                         **{c: round(v["dice"], 3) for c, v in mj["per_class"].items()
                            if v["present_in_ground_truth"] and v["dice"] is not None and c != "background"}})
if rows:
    st.subheader("Segmentation")
    st.dataframe(pd.DataFrame(rows), width="stretch")

# ---------------------------------------------------------------- segmentation training runs
with st.expander("Segmentation training histories"):
    for p in sorted(MODELS.glob("*/history.csv")):
        h = pd.read_csv(p).set_index("epoch")
        if "val_mean_dice" not in h:  # e.g. TipNet history (shown in its own section)
            continue
        st.write(f"**{p.parent.name}** - {len(h)} epochs, best val Dice {h['val_mean_dice'].max():.3f}")
        st.line_chart(h[["val_mean_dice"]])

if not os.environ.get("PRISM_MONITOR_ONCE"):
    time.sleep(refresh)
    st.rerun()
