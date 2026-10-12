"""Overnight autopilot steps for the DeepLabCut tip model (run by scripts/supervisor.py; no user interaction).

  python scripts/autopilot.py settle   # lower the learning rate and continue DLC v2 for 10 epochs (settling phase)
  python scripts/autopilot.py decide   # hybrid tests on the 4 human-labelled clips + choose the app default

Decision rule (fixed in advance): a DLC-based option replaces the current geometric default only if NEITHER tip gets
worse by more than 1 px median on the 40 human frames and the sum of the two medians improves. Only the hybrid source
can be an app default (DLC alone has no fallback when no DLC prediction file exists for a video).
Caveat recorded in the decision file: options are compared on the same 40 frames used for every earlier comparison.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY_DML = ROOT / ".venv-dml" / "Scripts" / "python.exe"
CLIPS = ("14859", "14939", "15019", "15099")
SETTLE_EPOCHS, SETTLE_LR = 10, 5e-5
BASE = {"lhook": 22.4, "grasper": 31.9}   # current default (hinge geometry), measured earlier on the same frames


def project() -> Path:
    return sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-MULTI16v2-auto-*"))[-1]


def train_dir() -> Path:
    return sorted(project().glob("dlc-models-pytorch/*/*/train"))[0]


def snaps() -> dict[int, Path]:
    out = {}
    for s in train_dir().glob("snapshot-*.pt"):
        m = re.search(r"(\d+)\.pt$", s.name)
        if m:
            out[int(m[1])] = s
    return out


KEEP = ROOT / "outputs" / "models" / "dlc_v2_keep"


def strip_optimizer(src: Path) -> Path:
    """Copy of a (CPU-converted) DLC snapshot WITHOUT optimizer/scheduler state. DLC restores the saved optimizer
    on resume, which silently brings back the old learning rate; stripping it makes the settle lr take effect."""
    import torch

    data = torch.load(str(src), map_location="cpu", weights_only=True)
    clean = {k: v for k, v in data.items() if k not in ("optimizer", "scheduler")}
    KEEP.mkdir(parents=True, exist_ok=True)
    dst = KEEP / f"settle_start_{src.stem}_noopt.pt"
    torch.save(clean, str(dst))
    return dst


def settle() -> int:
    cfg_p = train_dir() / "pytorch_config.yaml"
    bak = cfg_p.with_name("pytorch_config.before_settle.yaml")
    if not bak.exists():
        shutil.copy2(cfg_p, bak)
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    cfg["runner"]["optimizer"]["params"]["lr"] = SETTLE_LR
    cfg["runner"]["scheduler"]["params"] = {"lr_list": [[SETTLE_LR], [SETTLE_LR]], "milestones": [1000, 2000]}
    cfg["runner"]["snapshots"]["max_snapshots"] = 10
    cfg_p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    subprocess.run([str(PY_DML), "scripts/dlc_snapshots_to_cpu.py", str(project() / "dlc-models-pytorch")], cwd=ROOT,
                   capture_output=True)
    ep = max(snaps())
    log = LOGS / "settle_dlc.log"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} settling from epoch {ep}, lr {SETTLE_LR}\n")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000
    p = subprocess.Popen([str(PY_DML), "scripts/dlc_train_gpu.py", "--config", str(project() / "config.yaml"),
                          "--epochs", str(SETTLE_EPOCHS), "--batch-size", "8", "--save-epochs", "5",
                          "--snapshot", str(strip_optimizer(snaps()[ep]))], cwd=ROOT, stdout=open(log, "a", encoding="utf-8"),
                         stderr=subprocess.STDOUT, creationflags=flags)
    print(f"settling started pid {p.pid} from epoch {ep}")
    return 0


def eval_json(args: list[str]) -> dict:
    e = sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-eval-*"))[-1]
    r = subprocess.run([str(PY), "scripts/evaluate_dlc_heldout.py", "--eval-config", str(e / "config.yaml"),
                        "--exclude", "outputs/label_qa/excluded.json", "--min-conf", "0.0", *args],
                       cwd=ROOT, capture_output=True, text=True)
    t = r.stdout
    d = json.loads(t[t.index("{"):t.rindex("}") + 1])
    return {g: {"median_px": d[g]["median_px"], "recall": d[g]["detection_recall"], "pck20": d[g].get("pck@20px")}
            for g in ("lhook", "grasper")}


def decide() -> int:
    settle_dir = ROOT / "outputs" / "dlc" / "settle_model"
    pred_dir = settle_dir if list(settle_dir.glob("*DLC*.h5")) else ROOT / "outputs" / "dlc" / "multi16v2_model"
    if len(sys.argv) > 2:  # e.g. decide outputs/dlc/v3_model v3
        pred_dir = ROOT / sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    results = {"dlc_only": eval_json(["--pred-dir", str(pred_dir)])}
    variants = {"hybrid_all": None, "hybrid_grasper": ["grasper_tip", "grasper_tip_2"], "hybrid_lhook": ["lhook_tip"]}
    for name, slots in variants.items():
        runs = ROOT / "outputs" / f"runs_{name}{'_' + tag if tag else ''}"
        for c in CLIPS:
            pred = [h for h in pred_dir.glob(f"video01_{c}DLC*.h5") if "_meta" not in h.stem][0]
            ov = [f"keypoints.dlc_predictions={pred.as_posix()}", "keypoints.source=hybrid",
                  "instrument.checkpoint=outputs/models/instrument_seg/best.pt", "tissue.checkpoint=null"]
            if slots:
                ov.append("keypoints.hybrid={dlc_slots: [" + ", ".join(slots) + "]}")
            subprocess.run([str(PY), "run_pipeline.py", "--input", f"Videos/video01_{c}.mp4", "--out",
                            str(runs / f"video01_{c}"), *ov], cwd=ROOT, capture_output=True)
        results[name] = eval_json(["--pipeline-runs", str(runs)])
    results["geometry_current"] = {"lhook": {"median_px": BASE["lhook"]}, "grasper": {"median_px": BASE["grasper"]}}
    ok = {}
    for name in variants:
        r = results[name]
        lm, gm = r["lhook"]["median_px"], r["grasper"]["median_px"]
        if lm is None or gm is None:
            continue
        if lm <= BASE["lhook"] + 1.0 and gm <= BASE["grasper"] + 1.0 and lm + gm < sum(BASE.values()) \
                and r["lhook"]["recall"] >= 1.0 and r["grasper"]["recall"] >= 1.0:
            ok[name] = lm + gm
    choice = min(ok, key=ok.get) if ok else "geometry_current"
    decision = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dlc_predictions_from": str(pred_dir.relative_to(ROOT)),
                "results": results, "rule": "adopt a hybrid only if neither tip is >1 px worse, both recalls 100 %, "
                "and the sum of medians improves", "choice": choice,
                "caveat": "all options compared on the same 40 human frames (video01) used throughout; small set."}
    if choice != "geometry_current":
        cfg_p = ROOT / "configs" / "pipeline.yaml"
        shutil.copy2(cfg_p, cfg_p.with_name("pipeline.before_autopilot.yaml"))
        s = cfg_p.read_text(encoding="utf-8")
        s = re.sub(r"(?m)^  source: \w+", "  source: hybrid", s, count=1)
        slots = variants[choice]
        block = ("  # Autopilot " + decision["time"] + ": hybrid chosen on the 40 human frames "
                 f"({results[choice]['lhook']['median_px']:.1f} / {results[choice]['grasper']['median_px']:.1f} px vs "
                 f"{BASE['lhook']} / {BASE['grasper']}). Needs a DLC prediction file next to the video; else mask-tip.\n"
                 "  hybrid:\n    dlc_slots: " + (json.dumps(slots) if slots else "null") +
                 "\n    accept_conf: 0.6\n    on_mask_px: 12\n")
        s = s.replace("  points:\n", block + "  points:\n", 1)
        cfg_p.write_text(s, encoding="utf-8")
        decision["config_changed"] = "configs/pipeline.yaml (backup pipeline.before_autopilot.yaml)"
    (LOGS / f"autopilot_decision{'_' + tag if tag else ''}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit({"settle": settle, "decide": decide}[sys.argv[1]]())
