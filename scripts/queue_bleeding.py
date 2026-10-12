"""Start scripts/train_bleeding.py once the clean DLC evaluation chain has finished (one GPU job at a time)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import supervisor as s  # noqa: E402

log = s.LOGS / "clean_eval.log"
while "all done" not in (log.read_text(errors="replace") if log.exists() else ""):
    time.sleep(60)
while s.running("dlc_train_gpu"):
    time.sleep(60)
(s.LOGS / "bleeding_train.log").write_text(time.strftime("start %Y-%m-%d %H:%M:%S\n"))
pid = s.start([s.PY_DML, "scripts/train_bleeding.py", "--epochs", "25"], s.LOGS / "bleeding_train.log")
time.sleep(30)
s.gentle(s.running("train_bleeding"))
