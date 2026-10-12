"""Risk / near-miss event annotation tool for surgeons (blind to the PRISM score by default).

  .venv\\Scripts\\streamlit run src/ui/annotate_events.py --server.port 8504

Writes one CSV per annotator to outputs/annotations/risk_events_<annotator>.csv
(override the folder with the PRISM_ANNOTATION_DIR environment variable).
Format and definitions: configs/risk_events.yaml, docs/RISK_ANNOTATION.md.
After annotating a whole video, add a `review_complete` row covering the whole
video so the validator knows everything else in it is negative.
"""

import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import streamlit as st  # noqa: E402

from src.risk.labels import (annotation_path, append_annotation, delete_annotation_rows,  # noqa: E402
                             load_schema, marker_type, read_annotation_file)

OUT_DIR = Path(os.environ.get("PRISM_ANNOTATION_DIR", ROOT / "outputs" / "annotations"))
st.set_page_config(page_title="PRISM-Endo event annotation", layout="wide")
schema = load_schema()
st.title("Risk / near-miss event annotation")
if schema.get("status") != "confirmed":
    st.warning("Event definitions in configs/risk_events.yaml are a PROPOSAL until the supervising surgeon "
               "confirms them (set status: confirmed).")


def list_videos():
    vids = {p.stem: p for p in sorted((ROOT / "Videos").glob("*.mp4"))}
    for run in sorted((ROOT / "outputs" / "runs").glob("*")):
        for name in ("original_h264.mp4", "annotated_h264.mp4"):
            if (run / name).is_file():
                vids.setdefault(run.name, run / "original_h264.mp4" if (run / "original_h264.mp4").is_file()
                                else run / name)
    return vids


@st.cache_data
def video_meta(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def frame(path: str, idx: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None


with st.sidebar:
    annotator = st.text_input("Annotator ID (pseudonymous)", value=st.session_state.get("annotator", ""))
    st.session_state["annotator"] = annotator
    vids = list_videos()
    vid = st.selectbox("Video", list(vids)) if vids else None
    st.caption("Shows the ORIGINAL video (no PRISM overlay or score) so labels stay blind.")

if not vids:
    st.error("No videos found in Videos/ or outputs/runs/.")
    st.stop()
if not annotator:
    st.info("Enter your annotator ID in the sidebar to start.")
    st.stop()

path = str(vids[vid])
fps, n = video_meta(path)
idx = st.slider("Frame", 0, max(0, n - 1), 0)
t = idx / fps
img = frame(path, idx)
if img is not None:
    st.image(img, caption=f"{vid} - frame {idx} - t = {t:.2f} s", width="stretch")

c1, c2, c3 = st.columns(3)
if c1.button("Set START here"):
    st.session_state["start_s"] = round(t, 3)
if c2.button("Set END here"):
    st.session_state["end_s"] = round(t, 3)
start_s, end_s = st.session_state.get("start_s"), st.session_state.get("end_s")
c3.write(f"start = {start_s} s, end = {end_s} s")

types = [k for k in schema["event_types"] if k != marker_type(schema)]
etype = st.selectbox("Event type", types, format_func=lambda k: f"{k} - {schema['event_types'][k].get('definition', '')[:90]}")
severity = st.radio("Severity", list(schema["severity"]), horizontal=True, format_func=lambda k: f"{k}: {schema['severity'][k]}")
confidence = st.radio("Confidence", list(schema["confidence"]), horizontal=True,
                      format_func=lambda k: f"{k}: {schema['confidence'][k]}")
notes = st.text_input("Notes (required for 'other')")
out = annotation_path(OUT_DIR, annotator)

b1, b2 = st.columns(2)
if b1.button("Save event", type="primary"):
    if start_s is None or end_s is None or start_s >= end_s:
        st.error("Set START before END (start < end).")
    elif etype == "other" and not notes.strip():
        st.error("Notes are required for event type 'other'.")
    else:
        append_annotation(out, {"video_id": vid, "start_s": start_s, "end_s": end_s, "label": 1,
                                "event_type": etype, "severity": severity, "annotator": annotator,
                                "confidence": confidence, "notes": notes, "blind": True,
                                "created_at": dt.datetime.now().isoformat(timespec="seconds")})
        st.success(f"Saved to {out}")
        st.session_state.pop("start_s", None)
        st.session_state.pop("end_s", None)
if b2.button("Mark whole video reviewed"):
    append_annotation(out, {"video_id": vid, "start_s": 0.0, "end_s": round((n - 1) / fps, 3), "label": 0,
                            "event_type": marker_type(schema), "annotator": annotator, "blind": True,
                            "created_at": dt.datetime.now().isoformat(timespec="seconds")})
    st.success("Coverage marker saved: everything not marked in this video counts as negative.")

st.subheader(f"Your rows for {vid}")
df = read_annotation_file(out) if out.is_file() else None
if df is not None and len(df):
    mine = df[df["video_id"].astype(str) == vid]
    st.dataframe(mine, width="stretch")
    sel = st.multiselect("Delete rows (index)", list(mine.index))
    if sel and st.button("Delete selected"):
        delete_annotation_rows(out, sel)
        st.rerun()
else:
    st.caption("No rows yet.")
