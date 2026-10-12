"""Simple click-to-label tool for instrument tips (writes DeepLabCut CollectedData files).

  .venv\\Scripts\\streamlit run src/ui/label_tips.py --server.port 8601

One frame at a time. The ground-truth instrument SEGMENTATION outline is drawn to help find each
tool; YOU decide and click where the tip is. Nothing is labelled automatically. Saves
labeled-data/<clip>/CollectedData_<scorer>.h5/.csv exactly as DeepLabCut expects, plus
reviewed.json (which frames you have looked at, so "no tip visible" frames count as done).
Folder overridable with PRISM_DLC_PROJECTS (used by tests).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from streamlit_image_coordinates import streamlit_image_coordinates  # noqa: E402

PROJECTS = Path(os.environ.get("PRISM_DLC_PROJECTS", ROOT / "outputs" / "dlc_projects"))
COLORS = {"lhook_tip": (255, 0, 255), "grasper_tip": (0, 200, 255), "grasper_tip_2": (255, 160, 0)}  # RGB
HELP = {"lhook_tip": "L-hook: the far end of the metal hook (where the 'L' bends)",
        "grasper_tip": "Grasper whose shaft enters from the LEFT (or the only grasper): the JAW HINGE (where the jaws meet the shaft)",
        "grasper_tip_2": "Second grasper, shaft entering from the RIGHT: the JAW HINGE (where the jaws meet the shaft)"}

st.set_page_config(page_title="PRISM-Endo tip labelling", layout="wide")


def load_project(kind: str):
    proj = sorted(PROJECTS.glob(f"prism-tips-{kind}-*"))[-1]
    cfg = yaml.safe_load((proj / "config.yaml").read_text(encoding="utf-8"))
    man = pd.read_csv(proj / "selection_manifest.csv")
    frames = [(r.clip, r.image, int(r.cholecseg8k_frame), r.instruments_in_gt_mask) for r in man.itertuples()]
    frames.sort(key=lambda f: (f[0], int("".join(c for c in f[1] if c.isdigit()))))
    return proj, cfg, frames


def labels_path(proj, clip, scorer):
    return proj / "labeled-data" / clip / f"CollectedData_{scorer}.h5"


def read_labels(proj, clip, scorer, bodyparts):
    p = labels_path(proj, clip, scorer)
    if p.is_file():
        df = pd.read_hdf(p)
        return {(img[-1] if isinstance(img, tuple) else Path(img).name): {
            bp: (df.loc[img, (scorer, bp, "x")], df.loc[img, (scorer, bp, "y")]) for bp in bodyparts}
            for img in df.index}
    return {}


def write_labels(proj, clip, scorer, bodyparts, table):
    imgs = sorted(table)
    cols = pd.MultiIndex.from_product([[scorer], bodyparts, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    idx = pd.MultiIndex.from_tuples([("labeled-data", clip, im) for im in imgs])
    rows = [[v for bp in bodyparts for v in table[im].get(bp, (np.nan, np.nan))] for im in imgs]
    df = pd.DataFrame(rows, index=idx, columns=cols).astype(float)
    p = labels_path(proj, clip, scorer)
    df.to_hdf(p, key="df_with_missing", mode="w")
    df.to_csv(p.with_suffix(".csv"))


def reviewed_file(proj):
    return proj / "reviewed.json"


def load_reviewed(proj):
    f = reviewed_file(proj)
    return set(json.loads(f.read_text())) if f.is_file() else set()


def save_reviewed(proj, s):
    reviewed_file(proj).write_text(json.dumps(sorted(s)))


def draw(proj, clip, img_name, csk_frame, points, outline: bool = False):
    img = cv2.cvtColor(cv2.imread(str(proj / "labeled-data" / clip / img_name)), cv2.COLOR_BGR2RGB)
    video = clip.split("_")[0]  # e.g. video12_00160 -> video12
    for root in ((ROOT / "video01", ROOT / "data" / "cholecseg8k" / video) if outline else ()):
        m = root / clip / f"frame_{csk_frame}_endo_watershed_mask.png"
        if m.is_file():
            lab = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            for cid, col in ((32, (255, 0, 255)), (31, (0, 200, 255))):
                cnts, _ = cv2.findContours((lab == cid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, col, 1, cv2.LINE_AA)
            break
    for bp, (x, y) in points.items():
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, (int(x), int(y)), 7, COLORS[bp], 2, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 2, COLORS[bp], -1)
            cv2.putText(img, bp, (int(x) + 9, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[bp], 1, cv2.LINE_AA)
    return img


KINDS = {"GRASPER": "Grasper frames, 5 test videos (40)", "LHOOK": "L-hook frames, other videos (120)", "train": "Training frames (200)",
         "eval": "Held-out test frames (40)"}
kinds = [k for k in KINDS if list(PROJECTS.glob(f"prism-tips-{k}-*"))]
kind = st.sidebar.radio("Project", kinds, format_func=KINDS.get)
proj, cfg, frames = load_project(kind)
scorer, bodyparts = cfg["scorer"], list(cfg["bodyparts"])
to_label = [b for b in cfg.get("label_only", bodyparts) if b in bodyparts]  # e.g. L-hook-only project
reviewed = load_reviewed(proj)
key = f"i_{kind}"
if key not in st.session_state:
    st.session_state[key] = next((i for i, f in enumerate(frames) if f"{f[0]}/{f[1]}" not in reviewed), 0)
i = st.session_state[key]
clip, img_name, csk, present = frames[i]
fid = f"{clip}/{img_name}"
tables = st.session_state.setdefault(f"tables_{kind}", {})
if clip not in tables:
    tables[clip] = read_labels(proj, clip, scorer, bodyparts)
points = tables[clip].setdefault(img_name, {bp: (np.nan, np.nan) for bp in bodyparts})

st.sidebar.progress(len(reviewed) / len(frames), text=f"{len(reviewed)} / {len(frames)} frames done")
show_outline = st.sidebar.checkbox("Show tool outline (segmentation)", value=False,
                                   help="Off: you see the plain frame and judge the tip yourself.")
with st.sidebar.expander("How to label (read once)", expanded=len(reviewed) == 0):
    st.markdown(
        "1. Pick **which tip** you are placing (buttons above the picture).\n"
        "2. **Click the tip** on the picture: the end of the tool that points into the **middle of the "
        "picture**, away from where the tool enters at the edge. (An outline can be switched on in the sidebar.)\n"
        "3. Not visible, hidden or unsure (> ~10 px)? Press **Not visible** for that tip. Do not guess.\n"
        "4. Press **Save & next**. Every frame must be saved, even when no tip is visible.")

REF = ROOT / "outputs" / "endovis" / "gallery"
with st.sidebar.expander("Reference: expert-labelled examples (EndoVis)", expanded=len(reviewed) < 5):
    st.markdown(
        "Human expert labels from the EndoVis'15 dataset (robotic instruments, Du et al. 2018). Use them to see "
        "**how precisely** tips are placed:\n"
        "- **Left/RightClasper** = the two **jaw tips** → your **grasper tip** is the point **midway between "
        "the two jaw tips** (or the single tip when the jaws are closed).\n"
        "- **End/Shaft** points mark the shaft entering from the image edge; the tip is at the **opposite end**.\n"
        "- Your **L-hook tip** has no EndoVis equivalent: click the **far end of the metal hook**.")
    for p in sorted(REF.glob("train_seq*_frame*.jpg"))[:3]:
        st.image(str(p), caption=p.stem, width="stretch")

st.subheader(f"Frame {i + 1} of {len(frames)} · {clip} · {img_name}  ({'done' if fid in reviewed else 'not done'})")
st.caption(f"Tools in this frame (from the segmentation): **{present}**")
active = st.radio("Which tip are you placing?", to_label, horizontal=True,
                  format_func=lambda b: {"lhook_tip": "L-hook tip", "grasper_tip": "Grasper tip (left/only)",
                                         "grasper_tip_2": "2nd grasper tip (right)"}[b], key=f"bp_{kind}_{i}")
st.caption(HELP[active])

img = draw(proj, clip, img_name, csk, points, outline=show_outline)
click = streamlit_image_coordinates(img, key=f"img_{kind}_{i}_{len([p for p in points.values() if np.isfinite(p[0])])}")
if click and (click["x"], click["y"]) != st.session_state.get("last_click"):
    st.session_state["last_click"] = (click["x"], click["y"])
    scale = img.shape[1] / click.get("width", img.shape[1])
    points[active] = (click["x"] * scale, click["y"] * scale)
    st.rerun()

st.write("Placed: " + ", ".join(f"**{b}** = ({x:.0f}, {y:.0f})" if np.isfinite(x) else f"{b} = not visible"
                                for b, (x, y) in points.items() if b in to_label))
c = st.columns(4)
if c[0].button("Not visible (this tip)"):
    points[active] = (np.nan, np.nan)
    st.rerun()
if c[1].button("◀ Previous") and i > 0:
    st.session_state[key] = i - 1
    st.rerun()
if c[2].button("Save & next ▶", type="primary"):
    write_labels(proj, clip, scorer, bodyparts, tables[clip])
    reviewed.add(fid)
    save_reviewed(proj, reviewed)
    st.session_state[key] = min(i + 1, len(frames) - 1)
    st.session_state.pop("last_click", None)
    st.rerun()
if c[3].button("Skip ▶") and i < len(frames) - 1:
    st.session_state[key] = i + 1
    st.rerun()
