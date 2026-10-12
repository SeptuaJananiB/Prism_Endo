"""Leakage-free DLC evaluation chain (video52 + video55 never used anywhere in training or initialisation).

  stage 1: ImageNet-initialised DLC on automatic labels of 14 videos (prism-tips-CLEANs1-ex5255-*), 45 epochs
  stage 2: fine-tune on the human L-hook clicks of 4 videos (prism-tips-CLEANs2-ex5255-*), 25 epochs
           (same recipe as the deployed DLC v3)
  freeze : the LAST stage-2 checkpoint (no selection on test data)
  test   : 35 human L-hook clicks on video52/video55 (scripts/eval_lhook_fold.py) -> outputs/fold_eval_clean/

Hybrid parameters are those already in configs/pipeline.yaml (chosen before this test). Log: outputs/logs/clean_eval.log
  .venv\\Scripts\\python scripts\\run_clean_eval.py
"""

import glob
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY, PY_DML = ROOT / ".venv/Scripts/python.exe", ROOT / ".venv-dml/Scripts/python.exe"
LOG = ROOT / "outputs/logs/clean_eval.log"
S1 = Path(sorted(glob.glob(str(ROOT / "outputs/dlc_projects/prism-tips-CLEANs1-ex5255-*")))[-1])
S2 = Path(sorted(glob.glob(str(ROOT / "outputs/dlc_projects/prism-tips-CLEANs2-ex5255-*")))[-1])
CLIPS = ["video52_02746", "video52_02826", "video52_02906", "video55_00508", "video55_00588", "video55_00668"]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def train(cfg: Path, epochs: int, snapshot: Path | None):
    cmd = [str(PY_DML), "scripts/dlc_train_gpu.py", "--config", str(cfg), "--epochs", str(epochs), "--batch-size", "8",
           "--save-epochs", "5"] + (["--snapshot", str(snapshot)] if snapshot else [])
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        time.sleep(30)
        half = list(range(max(1, psutil.cpu_count() // 2)))
        for q in [psutil.Process(p.pid)] + psutil.Process(p.pid).children(recursive=True):
            try:
                q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                q.cpu_affinity(half)
            except psutil.Error:
                pass
        rc = p.wait()
    log(f"train exit code {rc} ({cfg.parent.name})")
    if rc != 0:
        raise SystemExit(f"training failed: {cfg}")


def last_snapshot(proj: Path) -> Path:
    t = Path(glob.glob(str(proj / "dlc-models-pytorch/*/*/train"))[0])
    snaps = {int(re.search(r"(\d+)\.pt$", s.name)[1]): s for s in t.glob("snapshot-*.pt")}
    return snaps[max(snaps)]


def main():
    log("start clean evaluation chain")
    if not list((S1 / "dlc-models-pytorch").rglob("snapshot-*045.pt")):
        prev = list((S1 / "dlc-models-pytorch").rglob("snapshot-*.pt"))
        if prev:  # interrupted (shutdown): continue from the last saved snapshot, same stage-1 settings
            subprocess.run([str(PY_DML), "scripts/dlc_snapshots_to_cpu.py", str(S1 / "dlc-models-pytorch")], cwd=ROOT,
                           capture_output=True)
            snap = last_snapshot(S1)
            done = int(re.search(r"(\d+)\.pt$", snap.name)[1])
            log(f"stage 1 resume from {snap.name} for {45 - done} epochs")
            train(S1 / "config.yaml", 45 - done, snap)
        else:
            train(S1 / "config.yaml", 45, None)
    subprocess.run([str(PY_DML), "scripts/dlc_snapshots_to_cpu.py", str(S1 / "dlc-models-pytorch")], cwd=ROOT,
                   capture_output=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    import autopilot  # strip_optimizer: drop saved optimizer so the stage-2 learning rate applies

    s1_last = last_snapshot(S1)
    start = autopilot.strip_optimizer(s1_last)
    log(f"stage 1 done: {s1_last.name} -> stage 2 init {start.name}")
    if not list((S2 / "dlc-models-pytorch").rglob("snapshot-*070.pt")):
        train(S2 / "config.yaml", 25, start)
    subprocess.run([str(PY_DML), "scripts/dlc_snapshots_to_cpu.py", str(S2 / "dlc-models-pytorch")], cwd=ROOT,
                   capture_output=True)
    final = last_snapshot(S2)
    log(f"FROZEN model: {final} (last epoch, not selected on test data)")
    # scratch project holding only the frozen checkpoint
    q = ROOT / "outputs/models/dlc_clean_ex5255"
    shutil.rmtree(q, ignore_errors=True)
    q.mkdir(parents=True)
    shutil.copy2(S2 / "config.yaml", q / "config.yaml")
    shutil.copytree(S2 / "training-datasets", q / "training-datasets")
    m = final.parents[1]
    rel = m.relative_to(S2)
    (q / rel / "train").mkdir(parents=True)
    shutil.copytree(m / "test", q / rel / "test")
    shutil.copy2(m / "train" / "pytorch_config.yaml", q / rel / "train")
    shutil.copy2(final, q / rel / "train")
    c = yaml.safe_load((q / "config.yaml").read_text())
    c["project_path"] = str(q)
    (q / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    out = ROOT / "outputs/dlc/clean_model"
    out.mkdir(parents=True, exist_ok=True)
    code = ("import deeplabcut, time; t=time.time(); deeplabcut.analyze_videos(r'%s', %r, shuffle=1, save_as_csv=True, "
            "destfolder=r'%s', device='cpu'); print('ANALYZE_SECONDS', round(time.time()-t,1))"
            % (q / "config.yaml", [str(ROOT / f"Videos/{c}.mp4") for c in CLIPS], out))
    r = subprocess.run([str(PY), "-c", code], cwd=ROOT, capture_output=True, text=True)
    log("analysis: " + (re.search(r"ANALYZE_SECONDS [0-9.]+", r.stdout or "") or [r.stderr[-300:]])[0])
    r = subprocess.run([str(PY), "scripts/eval_lhook_fold.py", "--dlc-dir", "outputs/dlc/clean_model",
                        "--out", "outputs/fold_eval_clean"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "outputs/logs/clean_eval_result.log").write_text(r.stdout + r.stderr)
    log("all done")


if __name__ == "__main__":
    main()
