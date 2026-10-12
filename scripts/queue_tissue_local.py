"""Wait for the clean DLC chain to finish (one heavy GPU job at a time), then run the local full-data tissue
fine-tune with the gallbladder guard. Log: outputs/logs/tissue_dsad_local.log"""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs/logs/clean_eval.log"
while not (LOG.is_file() and "all done" in LOG.read_text(encoding="utf-8", errors="replace")):
    time.sleep(60)
with open(ROOT / "outputs/logs/tissue_dsad_local.log", "a", encoding="utf-8") as f:
    subprocess.run([str(ROOT / ".venv-dml/Scripts/python.exe"), "scripts/train_tissue_dsad.py", "--epochs", "8",
                    "--lr", "5e-5", "--dsad-weight", "0.5", "--max-cholec-drop", "0.01",
                    "--out", "outputs/models/tissue_seg_dsad_local"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
