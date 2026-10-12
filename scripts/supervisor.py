"""PRISM-Endo supervisor: run every 5 minutes by a Windows scheduled task; restarts whatever has stopped.

Processes are started DETACHED (not children of the interactive session), so they survive its low-memory clean-up.
Each run (a few seconds):
  1. watchdog             -> start if not running
  2. app :8501, dashboard :8600 -> start if the port does not answer
  3. DLC v2 training      -> if it is not running and has not finished (no TRAINING_DONE in its log), resume from
                              the newest checkpoint for the remaining epochs (gentle mode: below-normal priority,
                              half of the CPU cores)
  4. final scoring        -> once training finished and the log has no "all done", run scripts/finish_multi16v2.sh
  5. notification         -> one Windows notification when "all done" first appears
Log: outputs/logs/supervisor.log.

Install:  schtasks /Create /SC MINUTE /MO 5 /TN "PRISM-Endo Supervisor" /TR "<.venv>\\Scripts\\pythonw.exe <this file>" /F
Remove:   schtasks /Delete /TN "PRISM-Endo Supervisor" /F
"""

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
SUPLOG = LOGS / "supervisor.log"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY_DML = ROOT / ".venv-dml" / "Scripts" / "python.exe"
STREAMLIT = ROOT / ".venv" / "Scripts" / "streamlit.exe"
BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
TOTAL_EPOCHS = 80
DLC_LOG = LOGS / "multi16v2_dlc.log"
FLAGS = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # CREATE_NO_WINDOW


def log(msg: str) -> None:
    with open(SUPLOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def running(fragment: str, python_only: bool = True) -> list[psutil.Process]:
    """Processes whose command line contains `fragment`. python_only: ignore shells/PowerShell that merely MENTION it."""
    out = []
    for p in psutil.process_iter(["cmdline", "name"]):
        try:
            if python_only and not (p.info["name"] or "").lower().startswith("python"):
                continue
            if fragment in " ".join(p.info["cmdline"] or []):
                out.append(p)
        except psutil.Error:
            pass
    return out


def start(cmd: list, logfile: Path, append: bool = True) -> int:
    f = open(logfile, "a" if append else "w", encoding="utf-8")
    p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=FLAGS, close_fds=True)
    return p.pid


def port_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def gentle(procs) -> None:
    half = list(range(max(1, psutil.cpu_count() // 2)))
    for p in procs:
        try:
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            p.cpu_affinity(half)
        except psutil.Error:
            pass


def notify(title: str, text: str) -> None:
    ps = ("Add-Type -AssemblyName System.Windows.Forms;$n=New-Object System.Windows.Forms.NotifyIcon;"
          "$n.Icon=[System.Drawing.SystemIcons]::Information;$n.Visible=$true;"
          f"$n.ShowBalloonTip(30000,'{title}','{text}',[System.Windows.Forms.ToolTipIcon]::Info);Start-Sleep 31;$n.Dispose()")
    subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps], creationflags=FLAGS)


def dlc_project() -> Path | None:
    hits = sorted((ROOT / "outputs" / "dlc_projects").glob("prism-tips-MULTI16v2-auto-*"))
    return hits[-1] if hits else None


def last_epoch(text: str) -> int:
    eps = [int(m) for m in re.findall(r"^Epoch (\d+)/\d+", text, re.M)]
    return max(eps) if eps else 0


STAGE = LOGS / "autopilot_stage.txt"
STAGES = {"v2": (LOGS / "multi16v2_dlc.log", 80, "scripts/finish_multi16v2.sh"),
          "settle": (LOGS / "settle_dlc.log", 90, "scripts/finish_settle.sh")}


def ensure_training(log_p: Path, total: int, proj: Path, strip: bool = False) -> str:
    """Keep DLC training alive until `total` epochs; run the finisher once training ended. Returns the log text."""
    text = log_p.read_text(encoding="utf-8", errors="replace") if log_p.is_file() else ""
    training = running("dlc_train_gpu")
    if training:
        gentle(training)
    elif text and "TRAINING_DONE" not in text and "train exit" not in text:
        train = next(iter(sorted(proj.glob("dlc-models-pytorch/*/*/train"))), None)
        snaps = {int(m[1]): s for s in train.glob("snapshot-*.pt") if (m := re.search(r"(\d+)\.pt$", s.name))}
        if snaps and total - max(snaps) > 0:
            ep = max(snaps)
            subprocess.run([str(PY_DML), "scripts/dlc_snapshots_to_cpu.py", str(proj / "dlc-models-pytorch")],
                           cwd=ROOT, capture_output=True, creationflags=0x08000000)
            snap = snaps[ep]
            if strip:  # settle stage: drop saved optimizer state so the low learning rate is not overridden
                sys.path.insert(0, str(ROOT / "scripts"))
                import autopilot
                snap = autopilot.strip_optimizer(snap)
            pid = start([PY_DML, "scripts/dlc_train_gpu.py", "--config", proj / "config.yaml", "--epochs",
                         total - ep, "--batch-size", 8, "--save-epochs", 5, "--snapshot", snap], log_p)
            log(f"DLC training not running (checkpoint epoch {ep}, log epoch {last_epoch(text)}) -> resumed for "
                f"{total - ep} epochs, pid {pid}")
            time.sleep(20)
            gentle(running("dlc_train_gpu"))
    return text


def ensure_finisher(text: str, script: str) -> None:
    if ("TRAINING_DONE" in text or "train exit" in text) and "all done" not in text and not running(Path(script).stem, python_only=False):
        log(f"training finished, scoring not done -> started {script} pid "
            f"{start([BASH, script], LOGS / 'finisher_stdout.log')}")


def set_stage(st: str) -> None:
    STAGE.write_text(st)
    log(f"stage -> {st}")


def report() -> None:
    d = json.loads((LOGS / "autopilot_decision.json").read_text())
    r = d["results"]
    line = lambda k: f"{r[k]['lhook']['median_px']:.1f}/{r[k]['grasper']['median_px']:.1f}" if r.get(k) and         r[k]["lhook"]["median_px"] is not None and r[k]["grasper"]["median_px"] is not None else "n/a"
    msg = (f"DLC done. Choice: {d['choice']}. L-hook/grasper px: DLC {line('dlc_only')}, hybrid {line('hybrid_all')}, "
           f"current 22.4/31.9")
    try:  # status board rows
        import pandas as pd
        st = ROOT / "status" / "data"
        m = pd.read_csv(st / "tip_methods.csv", dtype=str, keep_default_na=False)
        for name, label in (("dlc_only", "DLC final (v2 + settling)"), ("hybrid_all", "Hybrid (DLC on instrument, else geometry)"),
                            ("hybrid_grasper", "Hybrid, DLC for grasper only"), ("hybrid_lhook", "Hybrid, DLC for L-hook only")):
            if r.get(name):
                v = r[name]
                fmt = lambda g: "-" if v[g]["median_px"] is None else f"{v[g]['median_px']:.1f} px"
                m.loc[len(m)] = [str(len(m) + 1), label, "1,387 auto labels, 16 videos", f"{v['lhook']['recall']:.0%}",
                                 fmt("lhook"), f"{v['grasper']['recall']:.0%}", fmt("grasper"),
                                 "ADOPTED as app default" if d["choice"] == name else "measured overnight"]
        m.to_csv(st / "tip_methods.csv", index=False)
        c = pd.read_csv(st / "completed.csv", dtype=str, keep_default_na=False)
        c.loc[len(c)] = [str(len(c) + 1), "DLC", "Overnight autopilot: settling + hybrid test + decision",
                         f"Choice: {d['choice']} (outputs/logs/autopilot_decision.json)"]
        c.to_csv(st / "completed.csv", index=False)
        ip = pd.read_csv(st / "in_progress.csv", dtype=str, keep_default_na=False)
        ip = ip[~ip["Task"].str.startswith("DLC")]
        ip["No"] = [str(i + 1) for i in range(len(ip))]
        ip.to_csv(st / "in_progress.csv", index=False)
        subprocess.run([str(PY), "scripts/update_status_board.py"], cwd=ROOT, capture_output=True)
    except Exception as e:
        log(f"status board update failed: {e}")
    try:  # documentation: one appended section with the measured numbers
        rows = "\n".join(f"| {k} | {v['lhook']['median_px']} | {v['grasper']['median_px']} |" for k, v in r.items())
        with open(ROOT / "docs" / "DLC_ANNOTATION.md", "a", encoding="utf-8") as f:
            f.write(f"\n\n## Overnight autopilot result ({d['time']})\n\nDLC v2 (1,387 auto labels, 16 videos, "
                    f"epochs 31-80) + settling phase (low learning rate) and hybrid variants, scored on the 40 human "
                    f"frames (median px, no confidence cut-off):\n\n| Option | L-hook | Grasper |\n|---|---|---|\n{rows}\n\n"
                    f"Rule: {d['rule']}. **Choice: {d['choice']}.** {d.get('config_changed', 'App default unchanged.')}"
                    f"\nCaveat: {d['caveat']}\n")
    except Exception as e:
        log(f"doc update failed: {e}")
    notify("PRISM-Endo: overnight DLC finished", msg[:250])
    log("REPORT " + msg)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    # 1. watchdog
    if not running("scripts/watchdog.py") and not running("scripts\watchdog.py"):
        log(f"watchdog not running -> started pid {start([PY, 'scripts/watchdog.py', '--interval', '30'], LOGS / 'watchdog_stdout.log')}")
    # 2. web apps: the PRISM-Endo app (React + FastAPI) on 8510, training dashboard on 8600
    if not port_ok(8510) and not running("src.api.server:app"):
        pid = start([PY_DML, "-m", "uvicorn", "src.api.server:app", "--port", "8510", "--host", "127.0.0.1"],
                    LOGS / "web_api.log")
        log(f"port 8510 down -> started PRISM-Endo app pid {pid}")
    for port, page in ((8600, "src/ui/training_monitor.py"),):
        if not port_ok(port) and not running(page):
            pid = start([STREAMLIT, "run", page, "--server.port", str(port), "--server.headless", "true",
                         "--browser.gatherUsageStats", "false"], LOGS / f"streamlit_{port}.log")
            log(f"port {port} down -> started {page} pid {pid}")
    # 3. overnight DLC stages: v2 -> settle -> decide -> report -> done
    proj = dlc_project()
    stage = STAGE.read_text().strip() if STAGE.is_file() else "v2"
    hhmm = int(time.strftime("%H%M"))
    if proj is None or stage == "done":
        return 0
    if stage in STAGES:
        log_p, total, fin = STAGES[stage]
        text = ensure_training(log_p, total, proj, strip=(stage == "settle"))
        ensure_finisher(text, fin)
        if "all done" in text:
            if stage == "v2" and hhmm < 715:
                r = subprocess.run([str(PY), "scripts/autopilot.py", "settle"], cwd=ROOT, capture_output=True, text=True)
                log("settle: " + (r.stdout.strip() or r.stderr.strip()[-300:]))
                set_stage("settle")
            else:
                set_stage("decide")
        elif stage == "settle" and 815 <= hhmm < 1200:
            log("settling not finished by 08:15 -> deciding with what is available")
            set_stage("decide")
    elif stage == "decide":
        if (LOGS / "autopilot_decision.json").is_file():
            set_stage("report")
        elif not running("autopilot.py"):
            log(f"decide -> started pid {start([PY, 'scripts/autopilot.py', 'decide'], LOGS / 'autopilot_decide.log')}")
    elif stage == "report":
        report()
        set_stage("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never crash silently
        log(f"ERROR {type(e).__name__}: {e}")
        sys.exit(1)
