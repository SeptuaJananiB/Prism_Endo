"""PRISM-Endo web app server: REST API + the built React UI (web/dist).

Serves past runs (summary, per-frame results, videos, annotated frames), starts new pipeline runs in a background
thread with progress, and serves the built React app from web/dist.

  .venv-dml\\Scripts\\python -m uvicorn src.api.server:app --port 8510
"""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils.config import load_config, resolve_path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"
CFG_PATH = ROOT / "configs" / "pipeline.yaml"
DIST = ROOT / "web" / "dist"
COLS_PREFIX = ("timestamp_s", "risk_score", "risk_category", "bleeding_frac", "bleeding_detected",
               "cvs_c1", "cvs_c2", "cvs_c3", "cvs_achieved", "inst_grasper_frac", "inst_l_hook_frac",
               "risk_explanation")
COLS_SUFFIX = ("_prox_distance_px", "_speed_px_s", "_kp_status", "_kp_source", "_prox_inside", "_prox_target",
               "_risk_explanation")
UPLOADS = ROOT / "outputs" / "uploads"
_STATS: dict[str, tuple[float, dict]] = {}


def _stats(d: Path) -> dict:
    """Small per-run statistics for the library (cached by results.csv mtime)."""
    f = d / "results.csv"
    m = f.stat().st_mtime if f.is_file() else 0
    if d.name in _STATS and _STATS[d.name][0] == m:
        return _STATS[d.name][1]
    out = {}
    try:
        df = pd.read_csv(f, usecols=lambda c: c in ("risk_category", "bleeding_detected", "cvs_achieved"))
        n = max(1, len(df))
        cat = df.get("risk_category", pd.Series(dtype=str)).fillna("Unavailable")
        out = {"hi_pct": round(100 * float(cat.isin(["Medium", "High"]).sum()) / n, 1),
               "high_pct": round(100 * float(cat.eq("High").sum()) / n, 1),
               "bleeding_frames": int(df["bleeding_detected"].fillna(False).astype(bool).sum()) if "bleeding_detected" in df else None,
               "cvs_pct": round(100 * float(df["cvs_achieved"].fillna(False).astype(bool).mean()), 1) if "cvs_achieved" in df else None}
    except Exception:
        pass
    _STATS[d.name] = (m, out)
    return out

app = FastAPI(title="PRISM-Endo API")
JOBS: dict[str, dict] = {}
_PIPES: dict[str, object] = {}
_LOCK = threading.Lock()


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _run_dir(name: str) -> Path:
    d = (RUNS / name).resolve()
    if RUNS.resolve() not in d.parents or not (d / "summary.json").is_file():
        raise HTTPException(404, "run not found")
    return d


_TISSUE: dict[str, list] = {}


def _tissue_classes(ckpt: str) -> list[dict]:
    """Tissue classes of a checkpoint with their held-out reliability (cached)."""
    if ckpt in _TISSUE:
        return _TISSUE[ckpt]
    out = []
    p = resolve_path(ckpt)
    try:
        import torch

        from src.tissue.support import load_support, reliability

        names = torch.load(str(p), map_location="cpu", weights_only=True)["meta"]["class_names"]
        sup, met = load_support(p)
        for c in names:
            if c in ("background", "instrument"):
                continue
            r = reliability(sup, met, c)
            out.append({"name": c, "reliability": r.get("reliability", "unknown"),
                        "dice": r.get("test_dice")})
    except Exception:
        pass
    _TISSUE[ckpt] = out
    return out


@app.get("/api/meta")
def meta():
    base = load_config(CFG_PATH)
    inputs = []
    for folder in (ROOT / "Videos", ROOT / "outputs" / "uploads"):
        if folder.is_dir():
            inputs += [str(p.relative_to(ROOT)) for p in sorted(folder.iterdir())
                       if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm") and "_mp4v" not in p.stem]
    by = base["tissue"].get("by_surgery") or {}
    return {"surgery_types": list(by.keys()), "inputs": inputs,
            "tissues": {k: _tissue_classes(v) for k, v in by.items()},
            "defaults": {"targets": list(base.get("proximity", {}).get("target_classes", [])),
                         "show_all_tissue": bool(base.get("output", {}).get("show_all_tissue", False)),
                         "mask_alpha": float(base.get("output", {}).get("mask_alpha", 0.4))}}


@app.get("/api/runs")
def runs():
    out = []
    for s in sorted(RUNS.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            j = json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            continue
        tck = str(j.get("tissue_checkpoint") or "")
        out.append({"name": s.parent.name, "frames": j.get("frames_processed"),
                    "fps": (j.get("runtime") or {}).get("processing_fps"), "modified": s.stat().st_mtime,
                    "source": Path(str(j.get("input", s.parent.name))).name,
                    "surgery": ("Other" if "dsad" in tck else "Gallbladder") if tck else None,
                    **_stats(s.parent)})
    return out


@app.get("/api/runs/{name}")
def run_detail(name: str):
    d = _run_dir(name)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(d / "results.csv")
    cols = [c for c in df.columns if c in COLS_PREFIX or c.endswith(COLS_SUFFIX)]
    rows = df[cols].astype(object).where(df[cols].notna(), None)
    data = {c: [_clean(v) for v in rows[c].tolist()] for c in cols}
    return JSONResponse({"name": name, "summary": summary, "series": data,
                         "videos": {k: (f"/api/runs/{name}/video/{k}" if summary["outputs"].get(v) else None)
                                    for k, v in (("original", "original_video_h264"),
                                                 ("annotated", "annotated_video_h264"))}})


@app.get("/api/runs/{name}/video/{kind}")
def run_video(name: str, kind: str):
    d = _run_dir(name)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    key = {"original": "original_video_h264", "annotated": "annotated_video_h264"}.get(kind)
    p = summary["outputs"].get(key) if key else None
    if not p or not Path(p).is_file():
        raise HTTPException(404, "video not available")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/runs/{name}/frame/{idx}")
def run_frame(name: str, idx: int, w: int = 0, kind: str = "annotated"):
    d = _run_dir(name)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    p = summary["outputs"].get("annotated_video" if kind == "annotated" else "original_video_h264")
    if not p or not Path(p).is_file():
        raise HTTPException(404, "video not available")
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "frame not found")
    if w and w < img.shape[1]:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80 if w else 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{name}/file/{kind}")
def run_file(name: str, kind: str):
    d = _run_dir(name)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    p = {"csv": d / "results.csv", "summary": d / "summary.json",
         "video": summary["outputs"].get("annotated_video_h264")}.get(kind)
    if not p or not Path(p).is_file():
        raise HTTPException(404, "file not available")
    p = Path(p)
    return FileResponse(p, filename=f"{name}_{p.name}")


@app.get("/api/llm/status")
def llm_status():
    from src.api.explain import config

    c = config()
    return {"configured": c["configured"], "model": c["model"]}


@app.post("/api/runs/{name}/explain")
def run_explain(name: str, refresh: bool = False):
    """gpt-oss explanation of an analysis (numbers only are sent; cached per run)."""
    from src.api.explain import explain

    try:
        r = explain(_run_dir(name), refresh)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {k: r[k] for k in ("text", "model", "created", "cached")}


class Flag(BaseModel):
    t: float
    frame: int
    kind: str  # agree | wrong | important
    note: str = ""


@app.get("/api/runs/{name}/flags")
def get_flags(name: str):
    f = _run_dir(name) / "review_flags.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []


@app.put("/api/runs/{name}/flags")
def put_flags(name: str, flags: list[Flag]):
    """Reviewer flags are saved next to the run (human feedback; never mixed into model outputs)."""
    f = _run_dir(name) / "review_flags.json"
    f.write_text(json.dumps([x.model_dump() for x in flags], indent=2), encoding="utf-8")
    return {"saved": len(flags)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "video.mp4").name
    if Path(name).suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "unsupported video type")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / name
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    return {"source": str(dest.relative_to(ROOT))}


class RunRequest(BaseModel):
    source: str
    surgery: str | None = None
    keypoint_source: str = "hybrid"
    risk: bool = True
    max_frames: int = 0
    stride: int = 1
    targets: list[str] | None = None        # tissues to measure distance to
    show_all_tissue: bool | None = None     # outline every tissue class, not only the targets
    mask_alpha: float | None = None         # overlay strength 0..1


def _pipeline(cfg: dict):
    key = json.dumps(cfg, sort_keys=True)
    with _LOCK:
        if key not in _PIPES:
            from src.pipeline.pipeline import PrismPipeline

            _PIPES.clear()  # cheap: the heavy model weights are cached and reused inside the pipeline modules
            _PIPES[key] = PrismPipeline(cfg)
        return _PIPES[key]


from src.pipeline.pipeline import Stopped as Cancelled  # noqa: E402  (one exception for both stop paths)


def _finalise(tmp: Path, name: str) -> str:
    """Move a finished run from its temporary folder to outputs/runs/<name>, fixing the paths in summary.json.
    Reviewer flags of the run being replaced are kept (same video, same frame numbering)."""
    dest = RUNS / name
    flags = (dest / "review_flags.json").read_bytes() if (dest / "review_flags.json").is_file() else None
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    if flags is not None:
        (dest / "review_flags.json").write_bytes(flags)
    f = dest / "summary.json"
    summary = json.loads(f.read_text(encoding="utf-8"))
    summary["outputs"] = {k: (str(dest / Path(v).name) if isinstance(v, str) and v else v)
                          for k, v in (summary.get("outputs") or {}).items()}
    f.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return name


def _work(job_id: str, req: RunRequest):
    job = JOBS[job_id]
    tmp = RUNS / f".partial_{job_id}"
    try:
        base = load_config(CFG_PATH)
        cfg = json.loads(json.dumps(base))
        by = base["tissue"].get("by_surgery") or {}
        if req.surgery in by:
            cfg["tissue"]["checkpoint"] = by[req.surgery]
        cfg["keypoints"]["source"] = req.keypoint_source
        cfg["risk"]["enabled"] = req.risk
        cfg["input"]["max_frames"] = int(req.max_frames) or None
        cfg["input"]["stride"] = int(req.stride)
        if req.targets:
            cfg.setdefault("proximity", {})["target_classes"] = list(req.targets)
        if req.show_all_tissue is not None:
            cfg.setdefault("output", {})["show_all_tissue"] = bool(req.show_all_tissue)
        if req.mask_alpha is not None:
            cfg.setdefault("output", {})["mask_alpha"] = max(0.0, min(1.0, float(req.mask_alpha)))
        job.update(status="loading", phase="Loading models")
        pipe = _pipeline(cfg)
        pipe.should_stop = lambda: bool(job.get("cancel"))
        pipe.on_phase = lambda ph: job.update(phase=ph)
        job["status"] = "running"

        def progress(i, n):
            job.update(done=i, total=n or job.get("total"))
            if job.get("cancel"):
                raise Cancelled()

        if job.get("cancel"):
            raise Cancelled()
        summary = pipe.run(resolve_path(req.source), tmp, progress)
        name = _finalise(tmp, str(summary.get("video_id") or Path(req.source).stem))
        job.update(status="finished", run=name, finished=time.time())
    except Cancelled:
        job.update(status="cancelled", finished=time.time())
    except Exception as e:  # report to the UI instead of crashing the server
        job.update(status="failed", error=str(e))
    finally:
        if tmp.exists():  # stopped or failed: never leave partial results behind
            shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/run")
def start_run(req: RunRequest):
    busy = _active()
    if busy:
        raise HTTPException(409, f"Another analysis ({Path(busy['source']).name}) is still running. "
                                 "Wait for it to finish or stop it first.")
    if not resolve_path(req.source).exists():
        raise HTTPException(400, "input not found")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "status": "queued", "done": 0, "total": None, "source": req.source,
                    "started": time.time()}
    threading.Thread(target=_work, args=(job_id, req), daemon=True).start()
    return JOBS[job_id]


def _active():
    return next((j for j in JOBS.values() if j["status"] in ("queued", "loading", "running")), None)


@app.get("/api/active")
def active():
    """The running analysis, if any (so every open page can show and stop it)."""
    return _active() or {}


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j["status"] in ("queued", "loading", "running"):
        j["cancel"] = True
        j["status_note"] = "stopping"
    return j


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        root = DIST.resolve()
        f = (DIST / path).resolve()
        # only files inside web/dist; anything else (including ../ tricks) gets the app shell
        if path and f.is_file() and root in f.parents:
            return FileResponse(f)
        # the app shell must never be cached, or browsers keep loading an old build after updates
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})
