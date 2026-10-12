"""Resource watchdog for long training runs (the laptop shut down twice under sustained load).

Every --interval seconds it checks CPU load, free RAM, free commit memory and battery, and:
  * memory pressure  -> trims the working sets of PRISM-Endo python processes (Windows EmptyWorkingSet: pages move
                        to the standby list; nothing is killed) and lowers training priority;
  * sustained CPU >= --cpu-high % -> lowers training priority and restricts CPU training to half the cores
                        (less heat and better responsiveness);
  * on battery and <= --battery-low %  -> suspends training processes; they are resumed when power returns;
  * calm again       -> restores normal priority and all cores.
  * GPU cool-down duty cycle (--duty RUN,PAUSE seconds, default 240,45): training processes are suspended for PAUSE s
    after every RUN s of work, so the GPU/CPU never run flat out for long (the laptop switched off twice under
    sustained GPU load). While throttled (hot or low memory) the pause is doubled. --duty 0,0 disables it.
Temperature sensors need admin rights here, so sustained CPU load is used as the heat proxy.
Log: outputs/logs/watchdog.log

  .venv\\Scripts\\python scripts\\watchdog.py
"""

import argparse
import ctypes
import os
import time
from collections import deque
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs" / "watchdog.log"
TRAIN_MARKERS = ("dlc_train_gpu", "wf.train", "train_segmentation", "train_tissue_dsad", "train_bleeding")


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def project_procs():
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if p.pid == os.getpid() or "python" not in (p.info["name"] or "").lower():
                continue
            if "PRISM-Endo" in cmd or "PRISM-Endo" in (p.info["cwd"] or ""):
                out.append((p, cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def is_training(cmd: str) -> bool:
    return any(m in cmd for m in TRAIN_MARKERS)


def trim(p: psutil.Process) -> bool:
    try:
        h = ctypes.windll.kernel32.OpenProcess(0x0100 | 0x0400, False, p.pid)  # SET_QUOTA | QUERY_INFORMATION
        ok = bool(h) and bool(ctypes.windll.psapi.EmptyWorkingSet(h))
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
        return ok
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--cpu-high", type=float, default=92.0)
    ap.add_argument("--sustain", type=int, default=6, help="checks in a row above --cpu-high before throttling")
    ap.add_argument("--ram-low-gb", type=float, default=1.5)
    ap.add_argument("--commit-low-gb", type=float, default=3.0)
    ap.add_argument("--battery-low", type=int, default=30)
    ap.add_argument("--duty", default="240,45", help="RUN,PAUSE seconds cool-down cycle for training (0,0 = off)")
    a = ap.parse_args(argv)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ncpu = psutil.cpu_count()
    half = list(range(max(1, ncpu // 2)))
    history = deque(maxlen=a.sustain)
    throttled = suspended = False
    run_s, pause_s = (int(x) for x in a.duty.split(","))
    paused, phase_t = False, time.time()
    log(f"watchdog started: {ncpu} logical CPUs, thresholds cpu>={a.cpu_high}% x{a.sustain}, "
        f"RAM<{a.ram_low_gb}GB, commit<{a.commit_low_gb}GB, battery<={a.battery_low}%")
    while True:
        cpu = psutil.cpu_percent(interval=1)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        free_ram = vm.available / 2**30
        free_commit = (vm.available + sw.free) / 2**30
        bat = psutil.sensors_battery()
        history.append(cpu)
        procs = project_procs()
        train = [(p, c) for p, c in procs if is_training(c)]
        hot = len(history) == a.sustain and min(history) >= a.cpu_high
        mem_low = free_ram < a.ram_low_gb or free_commit < a.commit_low_gb
        on_low_battery = bat is not None and not bat.power_plugged and bat.percent <= a.battery_low

        if on_low_battery and not suspended:
            for p, _ in train:
                try:
                    p.suspend()
                except psutil.Error:
                    pass
            suspended = True
            log(f"BATTERY {bat.percent}% unplugged -> suspended {len(train)} training processes")
        elif suspended and bat is not None and bat.power_plugged:
            for p, _ in train:
                try:
                    p.resume()
                except psutil.Error:
                    pass
            suspended = False
            log("power restored -> resumed training processes")

        # GPU cool-down duty cycle (independent of the battery suspension above)
        if run_s and pause_s and not suspended:
            now, want_pause = time.time(), pause_s * (2 if throttled else 1)
            if not paused and train and now - phase_t >= run_s:
                for p, _ in train:
                    try:
                        p.suspend()
                    except psutil.Error:
                        pass
                paused, phase_t = True, now
                log(f"COOL-DOWN pause {want_pause}s ({len(train)} training procs, cpu {cpu:.0f}%, throttled={throttled})")
            elif paused and now - phase_t >= want_pause:
                for p, _ in train:
                    try:
                        p.resume()
                    except psutil.Error:
                        pass
                paused, phase_t = False, now
                log("cool-down over -> resumed training")
            elif not train:
                phase_t = now
        if mem_low:
            n = sum(trim(p) for p, _ in procs)
            log(f"MEMORY free RAM {free_ram:.1f}GB commit {free_commit:.1f}GB -> trimmed {n} working sets")
        if (hot or mem_low) and not throttled:
            for p, c in train:
                try:
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    if "dlc_train_gpu" not in c:  # CPU trainer: fewer cores = less heat
                        p.cpu_affinity(half)
                except psutil.Error:
                    pass
            throttled = True
            log(f"THROTTLE cpu {cpu:.0f}% (sustained={hot}) mem_low={mem_low} -> below-normal priority, "
                f"CPU training on {len(half)} cores")
        elif throttled and not hot and not mem_low and max(history) < a.cpu_high - 15:
            for p, _ in train:
                try:
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
                    p.cpu_affinity(list(range(ncpu)))
                except psutil.Error:
                    pass
            throttled = False
            log(f"calm (cpu {cpu:.0f}%) -> normal priority, all cores")
        if int(time.time()) % 300 < a.interval:
            log(f"status cpu {cpu:.0f}% RAM free {free_ram:.1f}GB commit free {free_commit:.1f}GB "
                f"battery {bat.percent if bat else '?'}% plugged={bat.power_plugged if bat else '?'} "
                f"training procs {len(train)} throttled={throttled}")
        time.sleep(max(1, min(a.interval, 15) - 1))


if __name__ == "__main__":
    main()
