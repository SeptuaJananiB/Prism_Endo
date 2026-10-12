"""End-to-end PRISM-Endo video pipeline (Stage G).

input video / image folder
  -> instrument segmentation         (optional: needs a checkpoint)
  -> keypoints (DLC file or mask-tip heuristic)
  -> motion & trajectory per keypoint
  -> tissue segmentation              (optional: needs a checkpoint)
  -> proximity per keypoint           (only if tissue is available)
  -> risk score per keypoint + frame  (rule-based, or learned if trained)
  -> annotated video + per-frame CSV + summary.json

Frames are streamed one at a time; nothing is accumulated except the CSV
rows. Every missing component is reported in `warnings` instead of being
silently replaced with invented values.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.motion.camera import CameraMotionEstimator
from src.motion.kinematics import KeypointMotionTracker
from src.segmentation.temporal import TemporalSegmenter
from src.pipeline.overlay import draw_frame
from src.proximity.proximity import ProximityEstimator, nearest
from src.risk.rule_based import RiskOutput, RuleBasedRiskScorer, combine
from src.segmentation.predictor import SegmentationPredictor
from src.tissue.tissue_segmenter import TissueSegmenter
from src.tracking.keypoints import slot_classes, slot_names, specs_from_config
from src.proximity.scale import mm_per_px
from src.tracking.sources import DLCKeypointSource, HybridKeypointSource, MaskTipKeypointSource, find_dlc_predictions
from src.utils.config import resolve_path
from src.utils.device import describe_hardware, get_logger, select_device
from src.utils.video_io import FrameSource, VideoWriter, to_browser_mp4

log = get_logger("prism.pipeline")
_MODELS: dict = {}  # loaded CVS / bleeding models, reused across pipelines with different cheap settings


class Stopped(Exception):
    """Raised when should_stop() turns true (user pressed Stop)."""


def _cached(key, factory):
    if key not in _MODELS:
        _MODELS[key] = factory()
    return _MODELS[key]


class PrismPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.warnings: list[str] = []
        self.device = select_device(cfg.get("device", "auto"))
        if cfg.get("cpu_threads"):  # measured on a Ryzen 9 5980HX: 16 threads ~16% faster forward than 8
            import torch
            torch.set_num_threads(int(cfg["cpu_threads"]))
        self.specs = specs_from_config(cfg["keypoints"]["points"])

        # Stage A: instrument segmentation
        self.inst = None
        ck = resolve_path(cfg["instrument"].get("checkpoint"))
        if ck and ck.is_file():
            self.inst = SegmentationPredictor.from_checkpoint(ck, self.device)
            self.inst.upsample = cfg.get("segmentation", {}).get("upsample", "bilinear")
            self._maybe_onnx(self.inst, ck)
        else:
            self._warn(f"Instrument checkpoint not found ({ck}). Instrument masks disabled; "
                       f"train with: python train_segmentation.py --config configs/instrument_seg.yaml")

        # Stage D: tissue segmentation
        tcfg = cfg.get("tissue", {})
        self.targets = list(cfg.get("proximity", {}).get("target_classes", []))
        self.tissue = TissueSegmenter(resolve_path(tcfg.get("checkpoint")), self.device, self.targets,
                                      allow_untrained_targets=cfg.get("proximity", {}).get(
                                          "allow_untrained_targets", False))
        if not self.tissue.available:
            self._warn(self.tissue.message + " Running in instrument-only mode; proximity unavailable.")
        for w in self.tissue.warnings:
            self._warn(w)
        if self.tissue.available:
            self.tissue.predictor.upsample = cfg.get("segmentation", {}).get("upsample", "bilinear")
            self._maybe_onnx(self.tissue.predictor, resolve_path(tcfg.get("checkpoint")))

        # Temporal smoothing (flicker reduction) and camera-motion estimation.
        ccfg = cfg.get("camera_motion", {})
        self.camera_enabled = bool(ccfg.get("enabled", False))
        self.smooth_mc = bool(ccfg.get("compensate_smoothing", False)) and self.camera_enabled
        self.prox = ProximityEstimator(self.targets, cfg.get("proximity", {}).get("min_area", 50))

        rcfg = cfg.get("risk", {})
        self.risk_enabled = bool(rcfg.get("enabled", True))
        self.learned = None
        if self.risk_enabled and rcfg.get("mode") == "learned":
            from src.risk.learned import LearnedRiskScorer

            self.learned = LearnedRiskScorer(resolve_path(rcfg.get("learned_model")))
        self.mm_cfg = (cfg.get("proximity", {}) or {}).get("approx_mm", {}) or {}
        ccfg = (cfg.get("safety", {}) or {}).get("cvs", {}) or {}
        self.cvs = None
        if ccfg.get("enabled"):
            ck = resolve_path(ccfg.get("checkpoint"))
            if ck is None or not ck.is_file():
                self._warn(f"safety.cvs enabled but checkpoint {ck} not found: CVS assessment disabled.")
            else:
                from src.safety.cvs import CVSAssessor

                self.cvs = _cached(("cvs", str(ck), ck.stat().st_mtime, json.dumps(ccfg, sort_keys=True, default=str)), lambda: CVSAssessor(ck, ccfg.get("every_s", 1.0), ccfg.get("height", 320), ccfg.get("width", 576),
                                       ccfg.get("threshold", 0.5), cfg.get("cpu_threads")))
        bcfg = (cfg.get("safety", {}) or {}).get("bleeding", {}) or {}
        self.bleeding = None
        if bcfg.get("enabled"):
            ck = resolve_path(bcfg.get("checkpoint"))
            if ck is None or not ck.is_file():
                self._warn(f"safety.bleeding enabled but checkpoint {ck} not found: bleeding detection disabled.")
            else:
                from src.safety.bleeding import BleedingDetector

                self.bleeding = _cached(("bleed", str(ck), ck.stat().st_mtime, str(self.device), json.dumps(bcfg, sort_keys=True, default=str)), lambda: BleedingDetector(ck, bcfg.get("every_s", 0.5), bcfg.get("min_frac", 0.002),
                                                 self.device if bcfg.get("use_gpu", True) else "cpu"))
        self.should_stop: Callable[[], bool] = lambda: False   # set by the UI server: Stop button
        self.on_phase: Callable[[str], None] = lambda _p: None  # set by the UI server: what is happening now
        self._init_warnings = list(self.warnings)

    def _maybe_onnx(self, predictor, ckpt) -> None:
        """segmentation.backend: onnx -> use model.onnx next to the checkpoint (CPU only)."""
        if self.cfg.get("segmentation", {}).get("backend", "torch") != "onnx" or self.device != "cpu":
            return
        onnx = Path(ckpt).with_name("model.onnx")
        if not onnx.is_file():
            self._warn(f"segmentation.backend=onnx but {onnx} is missing (run scripts/export_onnx.py); using PyTorch.")
            return
        predictor.use_onnx(onnx, self.cfg.get("cpu_threads"))

    def _warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)

    def _auto_analyze(self, video: Path, dlc_config: Path) -> Path | None:
        """Run the stored DeepLabCut model on `video` once (cached in outputs/dlc/auto_analyze/<video stem>/)."""
        out = resolve_path("outputs/dlc/auto_analyze") / video.stem
        hits = [h for h in out.glob(f"{video.stem}DLC*.h5") if "_meta" not in h.stem] if out.is_dir() else []
        if hits:
            return hits[0]
        if not dlc_config.is_file():
            self._warn(f"hybrid.dlc_config {dlc_config} not found: L-hook falls back to the mask-tip estimate.")
            return None
        try:
            import shutil
            import subprocess
            import sys

            out.mkdir(parents=True, exist_ok=True)
            log.info("Running DeepLabCut on %s (first time only; cached in %s)", video.name, out)
            self.on_phase(f"Running DeepLabCut on {video.name} (first time for this video, then cached)")
            # separate process so Stop can end it at once (an in-process DLC call cannot be interrupted)
            code = ("import sys, deeplabcut; deeplabcut.analyze_videos(sys.argv[1], [sys.argv[2]], shuffle=1, "
                    "save_as_csv=True, destfolder=sys.argv[3], device='cpu')")
            with open(out / "analyze.log", "w", encoding="utf-8") as lf:
                p = subprocess.Popen([sys.executable, "-c", code, str(dlc_config), str(video), str(out)],
                                     stdout=lf, stderr=subprocess.STDOUT)
                while p.poll() is None:
                    if self.should_stop():
                        p.kill()
                        p.wait()
                        shutil.rmtree(out, ignore_errors=True)  # never cache a half-finished analysis
                        raise Stopped()
                    time.sleep(0.3)
            if p.returncode != 0:
                raise RuntimeError(f"exit code {p.returncode}, see {out / 'analyze.log'}")
        except Stopped:
            raise
        except Exception as e:  # never break the pipeline: fall back to the mask-tip estimate
            self._warn(f"DeepLabCut analysis failed ({type(e).__name__}: {e}); using the mask-tip estimate.")
            return None
        hits = [h for h in out.glob(f"{video.stem}DLC*.h5") if "_meta" not in h.stem]
        return hits[0] if hits else None

    def _keypoint_source(self, src_path: Path):
        kcfg = self.cfg["keypoints"]
        mode = kcfg.get("source", "auto")
        dlc_file = resolve_path(kcfg.get("dlc_predictions"))
        if mode in ("auto", "dlc", "hybrid") and dlc_file is None and src_path.is_file():
            dlc_file = find_dlc_predictions(src_path)
        if mode == "dlc" and (dlc_file is None or not dlc_file.is_file()) and src_path.is_file():
            h = kcfg.get("hybrid", {}) or {}  # same stored model as hybrid mode, run once per video and cached
            if h.get("auto_analyze") and h.get("dlc_config"):
                dlc_file = self._auto_analyze(src_path, resolve_path(h["dlc_config"]))
            if dlc_file is not None and dlc_file.is_file():
                self._warn("keypoints.source=dlc: every tip comes from DeepLabCut. The stored model was trained on "
                           "human L-hook clicks; its grasper points are not validated (hybrid mode uses geometry "
                           "for the grasper).")
        if mode == "dlc" or (mode == "auto" and dlc_file is not None and dlc_file.is_file()):
            if dlc_file is None or not dlc_file.is_file():
                raise FileNotFoundError(
                    f"keypoints.source=dlc but no DLC prediction file was found for {src_path}. Run "
                    f"scripts/dlc_workflow.py analyze first or set keypoints.dlc_predictions.")
            if not src_path.is_file():
                self._warn("DLC predictions are indexed by video frame; with an image-folder input the "
                           "sequential frame index is used - make sure DLC analysed the same frames.")
            return DLCKeypointSource(self.specs, dlc_file, kcfg.get("dlc_min_confidence", 0.6))
        if self.inst is None:
            self._warn("No DLC predictions and no instrument model: keypoints unavailable.")
            return None
        if mode == "auto":
            self._warn("No trained DeepLabCut predictions found: using the mask-tip HEURISTIC for keypoints "
                       "(geometric estimate, not a trained keypoint model).")
        mask_src = MaskTipKeypointSource(self.specs, self.inst.class_names,
                                         kcfg.get("mask_tip_min_confidence", 0.3), kcfg.get("mask_tip_min_area", 300),
                                         landmarks=kcfg.get("mask_tip_landmarks"))
        if mode == "hybrid":
            h = kcfg.get("hybrid", {}) or {}
            if (dlc_file is None or not dlc_file.is_file()) and h.get("auto_analyze") and h.get("dlc_config") \
                    and src_path.is_file():
                dlc_file = self._auto_analyze(src_path, resolve_path(h["dlc_config"]))
            if dlc_file is None or not dlc_file.is_file():
                self._warn("keypoints.source=hybrid but no DeepLabCut predictions were found for this video: "
                           "using the mask-tip estimate only.")
                return mask_src
            return HybridKeypointSource(mask_src, DLCKeypointSource(self.specs, dlc_file, 0.0),
                                        h.get("dlc_slots"), float(h.get("accept_conf", 0.6)), int(h.get("on_mask_px", 12)))
        return mask_src

    def run(self, source: str | Path, out_dir: str | Path | None = None,
            progress: Callable[[int, int], None] | None = None) -> dict:
        cfg = self.cfg
        self.warnings = list(self._init_warnings)  # per-run warnings (instance may be reused by the UI)
        src_path = Path(source)
        icfg = cfg.get("input", {})
        self.on_phase("Preparing")
        if self.should_stop():
            raise Stopped()
        frames = FrameSource(src_path, fps=icfg.get("fps"), max_frames=icfg.get("max_frames"),
                             stride=icfg.get("stride", 1))
        video_id = src_path.stem if src_path.is_file() else src_path.name
        out_dir = Path(out_dir) if out_dir else resolve_path(cfg["output"]["dir"]) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        kp_source = self._keypoint_source(src_path)
        # cached detectors keep their last result between runs: start each run fresh
        for det, attrs in ((self.cvs, ("last", "last_t")), (self.bleeding, ("mask", "last_t"))):
            if det is not None:
                for a in attrs:
                    setattr(det, a, None)
        self.on_phase("Processing frames")
        mcfg = cfg.get("motion", {})
        kp_names = slot_names(self.specs)  # one slot per tracked instrument instance
        trackers = {n: KeypointMotionTracker(**{k: v for k, v in mcfg.items()}) for n in kp_names}
        rcfg = cfg.get("risk", {})
        scorers = {n: RuleBasedRiskScorer(rcfg, n) for n in kp_names}
        camera_est = CameraMotionEstimator() if self.camera_enabled else None
        # Fresh temporal state per run (built here so swapped-in models, e.g. in the UI or tests, are used).
        self.inst_t = TemporalSegmenter(self.inst, float(cfg["instrument"].get("temporal_alpha", 1.0)),
                                        self.smooth_mc) if self.inst is not None else None
        self.tissue_t = TemporalSegmenter(getattr(self.tissue, "predictor", None) or self.tissue,
                                          float(cfg.get("tissue", {}).get("temporal_alpha", 1.0)),
                                          self.smooth_mc) if self.tissue.available else None
        prev_inst = None
        use_cam_speed = bool(cfg.get("camera_motion", {}).get("compensate_speed", True))
        ocfg = cfg.get("output", {})
        writer = VideoWriter(out_dir / "annotated.mp4", frames.fps / frames.stride) if ocfg.get("write_video", True) else None
        orig_writer = VideoWriter(out_dir / "original.mp4", frames.fps / frames.stride) \
            if ocfg.get("write_original_video", False) else None
        tissue_draw = (self.tissue.class_names[1:] if ocfg.get("show_all_tissue") else self.targets) \
            if self.tissue.available else []
        tissue_draw = [c for c in tissue_draw if c not in self.tissue.ignore_classes and c != "instrument"]
        every_n = max(1, int(cfg.get("tissue", {}).get("every_n_frames", 1)))
        target_marks = {t: "*" for t, i in self.tissue.target_info.items() if i.get("reliability") != "evaluated"}

        csv_path = out_dir / "results.csv"
        rows_written, fieldnames, csv_f, csv_w = 0, None, None, None
        timing = {"instrument": [], "keypoints": [], "tissue": [], "post": [], "total": []}
        cat_counts: dict[str, int] = {}
        tissue_labels = None
        total = len(frames)
        t_start = time.perf_counter()
        try:
            for fr in frames:
                t0 = time.perf_counter()
                h, w = fr.image.shape[:2]
                diag = math.hypot(w, h)
                camera = camera_est.update(fr.image, prev_inst > 0 if prev_inst is not None else None) \
                    if camera_est is not None else None
                t_cam = time.perf_counter()
                inst_labels = self.inst_t.predict(fr.image, fr.timestamp_s, camera) if self.inst_t else None
                prev_inst = inst_labels
                t1 = time.perf_counter()
                dlc_index = fr.source_index if frames.kind == "video" else fr.index
                kps = kp_source(fr.image, dlc_index, inst_labels) if kp_source is not None else []
                t2 = time.perf_counter()
                if self.tissue.available and (fr.index % every_n == 0 or tissue_labels is None):
                    tissue_labels = self.tissue_t.predict(fr.image, fr.timestamp_s, camera)
                t3 = time.perf_counter()

                row = {"video_id": video_id, "frame": fr.index, "source_index": fr.source_index,
                       "timestamp_s": round(fr.timestamp_s, 4), "image_diag": round(diag, 2),
                       "keypoint_source": getattr(kp_source, "name", "none"),
                       "tissue_available": self.tissue.available}
                if camera_est is not None:
                    row.update(camera.as_dict())
                if inst_labels is not None:
                    for cid, name in enumerate(self.inst.class_names[1:], 1):
                        row[f"inst_{name}_frac"] = round(float((inst_labels == cid).mean()), 5)
                motions, proxs, trails, risks = {}, {}, {}, []
                for kp in kps:
                    mo = trackers[kp.name].update(fr.timestamp_s, kp.x, kp.y, kp.valid,
                                                  kp.status == "low_confidence",
                                                  camera if use_cam_speed else None)
                    motions[kp.name] = mo
                    trails[kp.name] = trackers[kp.name].trail_points()
                    use_xy = (mo.x_smooth, mo.y_smooth) if kp.valid and mo.x_smooth is not None else (None, None)
                    pres = self.prox(tissue_labels, self.tissue.class_names, *use_xy,
                                     velocity=(mo.vx_px_s, mo.vy_px_s))
                    near = nearest(pres)
                    proxs[kp.name] = near
                    if self.risk_enabled:
                        r = scorers[kp.name].update(fr.timestamp_s, kp, mo, near, self.tissue.available, diag)
                    else:
                        r = RiskOutput(kp.name, "disabled", "disabled")
                    risks.append(r)
                    row.update({f"{kp.name}_x": kp.x, f"{kp.name}_y": kp.y,
                                f"{kp.name}_confidence": round(kp.confidence, 4),
                                f"{kp.name}_kp_status": kp.status, f"{kp.name}_kp_source": kp.source})
                    row.update(mo.as_dict(f"{kp.name}_"))
                    for pr in pres:
                        row[f"{kp.name}_dist_{pr.target}_px"] = pr.distance_px
                        row[f"{kp.name}_inside_{pr.target}"] = pr.inside
                    # Always write the same columns (None when unavailable) so the CSV schema
                    # does not depend on what happened to be visible in the first frame.
                    row.update({f"{kp.name}_prox_target": near.target if near else (pres[0].status if pres else "none"),
                                f"{kp.name}_prox_distance_px": near.distance_px if near else None,
                                f"{kp.name}_prox_distance_norm": near.distance_norm if near else None,
                                f"{kp.name}_prox_inside": near.inside if near else None,
                                f"{kp.name}_prox_approach_speed_px_s": near.approach_speed_px_s if near else None,
                                f"{kp.name}_prox_reliability": (self.tissue.target_info.get(near.target, {})
                                                                .get("reliability") if near else None)})
                    row.update(r.as_dict(f"{kp.name}_risk_"))
                    if self.mm_cfg.get("enabled"):  # always write the columns (stable CSV schema)
                        for col in ("mm_per_px_approx", "prox_distance_mm_approx", "speed_mm_s_approx"):
                            row[f"{kp.name}_{col}"] = None
                    if self.mm_cfg.get("enabled") and inst_labels is not None and kp.valid:
                        cls = self.inst.class_names.index(slot_classes(self.specs).get(kp.name, ""))                             if slot_classes(self.specs).get(kp.name, "") in self.inst.class_names else None
                        s_mm = mm_per_px(inst_labels == cls, (kp.x, kp.y), float(self.mm_cfg.get("shaft_mm", 5.0)))                             if cls is not None else None
                        row[f"{kp.name}_mm_per_px_approx"] = round(s_mm, 4) if s_mm else None
                        d = row.get(f"{kp.name}_prox_distance_px")
                        row[f"{kp.name}_prox_distance_mm_approx"] = round(d * s_mm, 1) if (s_mm and d is not None) else None
                        sp = row.get(f"{kp.name}_speed_px_s")
                        row[f"{kp.name}_speed_mm_s_approx"] = round(sp * s_mm, 1) if (s_mm and sp is not None and sp == sp) else None
                if self.cvs is not None:
                    row.update(self.cvs.update(fr.image, fr.timestamp_s))
                if self.bleeding is not None:
                    row.update(self.bleeding.update(fr.image, fr.timestamp_s))
                frame_risk = combine(risks)
                row.update({"risk_score": frame_risk.score if frame_risk else None,
                            "risk_category": frame_risk.category if frame_risk else "Unavailable",
                            "risk_mode": frame_risk.mode if frame_risk else
                            ("disabled" if not self.risk_enabled else "unavailable"),
                            "risk_keypoint": frame_risk.keypoint if frame_risk else "",
                            "risk_explanation": frame_risk.explanation if frame_risk else
                            "; ".join(r.explanation for r in risks)})
                if self.learned is not None:
                    row["learned_risk_prob"] = round(self.learned.update(row, kp_names), 4)
                cat_counts[row["risk_category"]] = cat_counts.get(row["risk_category"], 0) + 1

                if writer is not None:
                    status = [f"t={fr.timestamp_s:6.2f}s  kp={getattr(kp_source, 'name', 'none')}  "
                              f"tissue={'on' if self.tissue.available else 'OFF'}  dist/speed in image px",
                              *([f"CVS (safe-to-clip estimate) C1 {row['cvs_c1']:.2f} C2 {row['cvs_c2']:.2f} "
                                 f"C3 {row['cvs_c3']:.2f} -> {'ACHIEVED' if row['cvs_achieved'] else 'not achieved'}"]
                                if "cvs_c1" in row else []),
                              ""
                              + ("  (* = target not validated on held-out data)" if target_marks else "")]
                    vis = draw_frame(fr.image, inst_labels=inst_labels,
                                     inst_names=self.inst.class_names if self.inst else None,
                                     tissue_labels=tissue_labels, tissue_names=self.tissue.class_names,
                                     tissue_draw=tissue_draw, keypoints=kps, motions=motions, trails=trails,
                                     proximities=proxs, risks=risks, frame_risk=frame_risk,
                                     status_lines=status, alpha=ocfg.get("mask_alpha", 0.4),
                                     kp_class=slot_classes(self.specs),
                                     target_marks=target_marks)
                    if self.bleeding is not None:
                        from src.safety.bleeding import draw_bleeding

                        vis = draw_bleeding(vis, self.bleeding.mask, row.get("bleeding_detected", False))
                    writer.write(vis)
                if orig_writer is not None:
                    orig_writer.write(fr.image)
                t4 = time.perf_counter()
                row.update({"ms_camera": round(1000 * (t_cam - t0), 1),
                            "ms_instrument": round(1000 * (t1 - t_cam), 1), "ms_keypoints": round(1000 * (t2 - t1), 1),
                            "ms_tissue": round(1000 * (t3 - t2), 1), "ms_total": round(1000 * (t4 - t0), 1)})
                for k, v in zip(timing, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                    timing[k].append(v)

                if csv_w is None:
                    fieldnames = list(row)
                    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
                    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
                    csv_w.writeheader()
                extra = set(row) - set(fieldnames)
                if extra:  # schema is meant to be fixed; never drop data silently
                    raise RuntimeError(f"Internal error: CSV columns appeared after frame 0: {sorted(extra)}")
                csv_w.writerow(row)
                rows_written += 1
                if progress:
                    progress(rows_written, total)
        finally:
            if writer is not None:
                writer.release()
            if orig_writer is not None:
                orig_writer.release()
            if csv_f is not None:
                csv_f.close()

        wall = time.perf_counter() - t_start
        summary = {
            "input": str(src_path), "video_id": video_id, "frames_processed": rows_written,
            "source_fps": frames.fps, "stride": frames.stride, "device": self.device,
            "instrument_model": bool(self.inst), "tissue_model": self.tissue.available,
            "tissue_checkpoint": str((self.cfg.get("tissue", {}) or {}).get("checkpoint") or ""),
            "bleeding_detector": self.bleeding is not None, "cvs_model": self.cvs is not None,
            "keypoint_source": getattr(kp_source, "name", "none"),
            "proximity_targets": self.tissue.target_info,
            "risk_mode": (rcfg.get("mode") if self.risk_enabled else "disabled"),
            "risk_category_counts": cat_counts,
            "runtime": {"wall_seconds": round(wall, 2),
                        "processing_fps": round(rows_written / wall, 2) if wall else None,
                        **{f"mean_ms_{k}": round(1000 * float(np.mean(v)), 1) for k, v in timing.items() if v}},
            "hardware": describe_hardware(), "warnings": self.warnings,
            "outputs": {"csv": str(csv_path)},
            "disclaimer": "Research prototype. Distances/speeds are image-space (px). The risk score is a "
                          "heuristic research score, not a validated probability of injury.",
        }
        if writer is not None:
            summary["outputs"]["annotated_video"] = str(out_dir / "annotated.mp4")
            if ocfg.get("browser_playable", True):
                self.on_phase("Converting videos for the browser")
                h264 = to_browser_mp4(out_dir / "annotated.mp4", out_dir / "annotated_h264.mp4")
                summary["outputs"]["annotated_video_h264"] = str(h264) if h264 else None
        if orig_writer is not None:
            h264 = to_browser_mp4(out_dir / "original.mp4", out_dir / "original_h264.mp4")
            summary["outputs"]["original_video_h264"] = str(h264) if h264 else None
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Processed %d frames in %.1fs (%.2f fps) -> %s", rows_written, wall,
                 rows_written / wall if wall else 0, out_dir)
        return summary
