"""PRISM-Endo Streamlit interface (LEGACY - the main app is the React web app, src/api/server.py on :8510).

Start (from the project root):
  .venv-dml\\Scripts\\streamlit run src/ui/app.py      # AMD/Intel GPU via DirectML
  .venv\\Scripts\\streamlit run src/ui/app.py          # CPU (or CUDA if available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.utils.config import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="PRISM-Endo", layout="wide")
CFG_PATH = ROOT / "configs" / "pipeline.yaml"
UPLOADS = ROOT / "outputs" / "uploads"
RUNS = ROOT / "outputs" / "runs"


def list_inputs() -> list[str]:
    items = []
    for folder in (ROOT / "Videos", ROOT / "outputs" / "synthetic", UPLOADS):
        if folder.is_dir():
            items += [str(p.relative_to(ROOT)) for p in sorted(folder.glob("*.mp4")) if "_mp4v" not in p.stem]
    data = ROOT / "video01"
    if data.is_dir():
        items += [str(p.relative_to(ROOT)) for p in sorted(data.iterdir()) if p.is_dir()]
    return items


def checkpoint_classes(path: str | None) -> list[str]:
    p = resolve_path(path) if path else None
    if not p or not p.is_file():
        return []
    import torch

    try:
        meta = torch.load(str(p), map_location="cpu", weights_only=True)["meta"]
        return list(meta["class_names"])
    except Exception:
        return []


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline(cfg_json: str):
    from src.pipeline.pipeline import PrismPipeline

    return PrismPipeline(json.loads(cfg_json))


def read_frame(video: Path, idx: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None


st.title("PRISM-Endo")
st.caption("Predictive Risk Intelligence System for Endoscopic Surgery - **research prototype, not clinically "
           "validated**. Distances and speeds are image-space pixels, not millimetres. The risk score is a "
           "rule-based heuristic, not a probability of injury.")

base = load_config(CFG_PATH)
with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Existing video / clip", "Upload video", "Previous run"], label_visibility="collapsed")
    source = None
    if mode == "Existing video / clip":
        opts = list_inputs()
        source = st.selectbox("Video or CholecSeg8k clip folder", opts) if opts else None
        if not opts:
            st.info("No videos found. Create some with scripts/make_video_from_frames.py --all video01")
    elif mode == "Upload video":
        up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up is not None:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            dest = UPLOADS / Path(up.name).name
            dest.write_bytes(up.getbuffer())
            source = str(dest.relative_to(ROOT))
    else:
        runs = sorted([p.parent.name for p in RUNS.glob("*/summary.json")]) if RUNS.is_dir() else []
        prev = st.selectbox("Run", runs) if runs else None
        if prev:
            st.session_state["run_dir"] = str(RUNS / prev)

    st.header("Models")
    device = st.selectbox("Device", ["auto", "cpu", "dml", "cuda"], help="auto: CUDA > DirectML (AMD/Intel) > CPU")
    inst_ck = st.text_input("Instrument checkpoint", base["instrument"]["checkpoint"] or "")
    st.caption("found" if inst_ck and resolve_path(inst_ck).is_file() else "missing - instrument masks disabled")
    by_surgery = base["tissue"].get("by_surgery") or {}
    surgery = st.selectbox("Surgery type", list(by_surgery) or ["cholecystectomy"],
                           help="Picks the tissue model. Gallbladder: CholecSeg8k model (best there). Other: CholecSeg8k"
                                " + DSAD model - far fewer false 'liver' on appendix/bowel, slightly worse on gallbladder.")
    tis_ck = st.text_input("Tissue checkpoint", by_surgery.get(surgery) or base["tissue"]["checkpoint"] or "",
                           key=f"tis_ck_{surgery}")
    tis_ok = bool(tis_ck) and resolve_path(tis_ck).is_file()
    st.caption("found" if tis_ok else "missing - instrument-only mode, proximity unavailable")
    tissue_classes = [c for c in checkpoint_classes(tis_ck) if c not in ("background", "instrument")]
    rel = {}
    if tis_ok:
        from src.tissue.support import load_support, reliability

        sup, met = load_support(resolve_path(tis_ck))
        rel = {c: reliability(sup, met, c)["reliability"] for c in tissue_classes}
    badge = {"evaluated": "tested", "insufficient_test_data": "barely tested", "unevaluated": "NOT tested",
             "failed_on_heldout": "FAILED test", "untrained": "not trained"}
    targets = st.multiselect("Proximity targets", tissue_classes,
                             default=[t for t in base["proximity"]["target_classes"] if t in tissue_classes],
                             format_func=lambda c: f"{c} ({badge.get(rel.get(c), 'unknown')})" if rel else c,
                             disabled=not tis_ok,
                             help="'tested' = accuracy measured on unseen patients. Other targets give proximity/risk "
                                  "that cannot be trusted.")

    st.header("Keypoints")
    kp_source = st.selectbox("Keypoint source", ["hybrid", "dlc", "mask_tip", "auto"],
                             help="hybrid (recommended): DeepLabCut for the L-hook (runs the stored model once per "
                                  "video, cached), geometry for the grasper. auto: only uses DLC files found next to "
                                  "the video.")
    dlc_file = st.text_input("DLC predictions file (optional)", "")

    st.header("Risk")
    risk_on = st.checkbox("Enable risk scoring", value=True)
    no_tissue = st.selectbox("Without tissue model", ["motion_only", "disable"])
    max_frames = st.number_input("Max frames (0 = all)", 0, 100000, 0, step=25)
    stride = st.number_input("Process every Nth frame", 1, 25, 1)
    run_clicked = st.button("Run pipeline", type="primary", disabled=source is None)

if run_clicked and source:
    cfg = json.loads(json.dumps(base))
    cfg["device"] = device
    cfg["instrument"]["checkpoint"] = inst_ck or None
    cfg["tissue"]["checkpoint"] = tis_ck or None
    if targets:
        cfg["proximity"]["target_classes"] = targets
    cfg["keypoints"]["source"] = kp_source
    cfg["keypoints"]["dlc_predictions"] = dlc_file or None
    cfg["risk"]["enabled"] = risk_on
    cfg["risk"]["when_proximity_unavailable"] = no_tissue
    cfg["input"]["max_frames"] = int(max_frames) or None
    cfg["input"]["stride"] = int(stride)
    try:
        pipe = get_pipeline(json.dumps(cfg, sort_keys=True))
        bar = st.progress(0.0, text="Processing...")

        def progress(i, n):
            bar.progress(min(1.0, i / n) if n else 0.0, text=f"Frame {i}/{n or '?'}")

        summary = pipe.run(resolve_path(source), None, progress)
        bar.empty()
        st.session_state["run_dir"] = str(Path(summary["outputs"]["csv"]).parent)
    except Exception as e:  # show errors in the UI instead of a stack trace
        st.error(f"Pipeline failed: {e}")

run_dir = Path(st.session_state["run_dir"]) if st.session_state.get("run_dir") else None
if run_dir and (run_dir / "summary.json").is_file():
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(run_dir / "results.csv")
    warns = summary.get("warnings", [])
    if warns:
        with st.expander(f"{len(warns)} reliability notice(s) for this run - click to read", expanded=False):
            for w in warns:
                st.warning(w)
    rt = summary.get("runtime", {})
    c = st.columns(5)
    c[0].metric("Frames", summary["frames_processed"])
    c[1].metric("Processing FPS", rt.get("processing_fps"))
    c[2].metric("Keypoints", summary["keypoint_source"])
    c[3].metric("Tissue model", "on" if summary["tissue_model"] else "off")
    c[4].metric("Device", summary.get("device", "?"))

    v1, v2 = st.columns(2)
    orig = summary["outputs"].get("original_video_h264")
    ann = summary["outputs"].get("annotated_video_h264") or summary["outputs"].get("annotated_video")
    with v1:
        st.subheader("Original")
        if orig and Path(orig).is_file():
            st.video(orig)
        else:
            st.info("Original preview not available.")
    with v2:
        st.subheader("Annotated")
        if ann and Path(ann).is_file():
            st.video(ann)

    st.subheader("Overview")
    from src.ui.overview import risk_strip

    st.markdown("**Risk timeline** (rule-based estimate; hover for times)")
    st.markdown(risk_strip(df), unsafe_allow_html=True)

    st.subheader("Time series")
    t = df.set_index("timestamp_s")
    if "bleeding_detected" in df:
        nb = int(df["bleeding_detected"].fillna(False).astype(bool).sum())
        st.metric("Bleeding flagged", f"{nb} / {len(df)} frames ({100 * nb / max(1, len(df)):.0f} %)",
                  help="SurgBlood-trained detector, held-out F1 0.937 - research estimate, not clinically validated")
        if nb:
            st.area_chart(t[["bleeding_frac"]] if "bleeding_frac" in df else None)
    tabs = st.tabs(["Risk score", "Distance to tissue (px)", "Tip speed (px/s)", "Risk categories",
                    "Critical View of Safety"])
    with tabs[0]:
        if df["risk_score"].notna().any():
            st.line_chart(t[["risk_score"]])
        else:
            st.info("No risk score (disabled, or no valid keypoints).")
    with tabs[1]:
        cols = [c for c in df.columns if c.endswith("_prox_distance_px")]
        if cols and df[cols].notna().any().any():
            st.line_chart(t[cols])
        else:
            st.info("Proximity unavailable (no tissue model, no target visible, or no valid keypoint).")
    with tabs[2]:
        cols = [c for c in df.columns if c.endswith("_speed_px_s")]
        if cols:  # a bare `x if c else y` expression would be echoed onto the page by Streamlit "magic"
            st.line_chart(t[cols])
        else:
            st.info("No motion data.")
    with tabs[3]:
        st.bar_chart(df["risk_category"].fillna("Unavailable").value_counts())
    with tabs[4]:
        cvs_cols = [c for c in ("cvs_c1", "cvs_c2", "cvs_c3") if c in df.columns]
        if cvs_cols:
            st.caption("Probability that each Critical View of Safety criterion is met (C1 hepatocystic triangle "
                       "cleared, C2 lower gallbladder off the cystic plate, C3 two structures to the gallbladder). "
                       "Learned from Endoscapes surgeon labels; test mAP 0.539 on unseen videos. A 'safe to clip?' "
                       "estimate - not a tissue-injury risk, not clinically validated.")
            st.line_chart(t[cvs_cols])
            st.metric("Frames with all 3 criteria met", f"{df['cvs_achieved'].mean():.0%}")
        else:
            st.info("CVS assessment not in this run (enable safety.cvs in the config).")

    st.subheader("Frame inspector")
    idx = st.slider("Frame", 0, max(0, len(df) - 1), 0)
    raw_ann = Path(summary["outputs"].get("annotated_video", ""))
    if raw_ann.is_file():
        img = read_frame(raw_ann, idx)
        if img is not None:
            st.image(img, width="stretch")
    row = df.iloc[idx]
    st.write(f"**t = {row['timestamp_s']:.2f} s** - category **{row['risk_category']}** - {row['risk_explanation']}")
    with st.expander("All values for this frame"):
        st.dataframe(row.astype(str).to_frame("value"), width="stretch")  # mixed types -> text

    st.subheader("Export")
    e = st.columns(3)
    e[0].download_button("results.csv", (run_dir / "results.csv").read_bytes(), "results.csv", "text/csv")
    e[1].download_button("summary.json", (run_dir / "summary.json").read_bytes(), "summary.json", "application/json")
    if ann and Path(ann).is_file():
        e[2].download_button("annotated video", Path(ann).read_bytes(), Path(ann).name, "video/mp4")
else:
    st.info("Choose an input in the sidebar and press **Run pipeline**, or open a previous run.")
