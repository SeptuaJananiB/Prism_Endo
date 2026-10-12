#!/usr/bin/env bash
# Post-training steps for the DLC settling phase (epochs 81-90, low learning rate).
# Waits until the training process has finished, then converts snapshots, analyses the 4 test clips with the final
# snapshot and scores on the 40 HUMAN frames.
set -e
cd "$(dirname "$0")/.."
P=$(ls -d outputs/dlc_projects/prism-tips-MULTI16v2-auto-* | tail -1)
LOG=outputs/logs/settle_dlc.log
until ! powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like '*dlc_train_gpu*MULTI16v2*' }" | grep -q .; do sleep 60; done
echo "train exit (detected) $(date '+%H:%M:%S')" >> $LOG
.venv-dml/Scripts/python scripts/dlc_snapshots_to_cpu.py $P/dlc-models-pytorch >> $LOG 2>&1
# Analyse with EXACTLY the newest numbered checkpoint: copy it alone into a scratch project (DLC's own snapshot
# index also counts the "best" checkpoint, which made index -1 / n-1 pick the wrong epoch).
.venv/Scripts/python - >> $LOG 2>&1 <<PY
import glob, re, shutil, yaml, deeplabcut
from pathlib import Path
P = Path("$P"); T = Path(glob.glob(str(P / "dlc-models-pytorch/*/*/train"))[0]); M = T.parent
snaps = {int(re.search(r"(\d+)\.pt$", s.name)[1]): s for s in T.glob("snapshot-*.pt")}  # numbered or "best" (DLC may name the final epoch snapshot-best-NNN)
ep = max(snaps); Q = Path("outputs/dlc/_scratch_settle_project")
shutil.rmtree(Q, ignore_errors=True); Q.mkdir(parents=True)
shutil.copy2(P / "config.yaml", Q / "config.yaml"); shutil.copytree(P / "training-datasets", Q / "training-datasets")
R = M.relative_to(P); (Q / R / "train").mkdir(parents=True); shutil.copytree(M / "test", Q / R / "test")
shutil.copy2(T / "pytorch_config.yaml", Q / R / "train"); shutil.copy2(snaps[ep], Q / R / "train")
c = yaml.safe_load((Q / "config.yaml").read_text()); c["project_path"] = str(Q.resolve())
(Q / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
print("analysing with", snaps[ep].name)
vids = [f"Videos/video01_{v}.mp4" for v in (14859, 14939, 15019, 15099)]
deeplabcut.analyze_videos(str(Q / "config.yaml"), vids, shuffle=1, save_as_csv=True, destfolder="outputs/dlc/settle_model", device="cpu")
PY
E=$(ls -d outputs/dlc_projects/prism-tips-eval-* | tail -1)
for c in 0.5 0.1 0.0; do
  .venv/Scripts/python scripts/evaluate_dlc_heldout.py --eval-config $E/config.yaml --pred-dir outputs/dlc/settle_model \
    --exclude outputs/label_qa/excluded.json --min-conf $c > outputs/logs/settle_dlc_eval_c$c.log 2>&1
done
echo "all done $(date '+%H:%M:%S')" >> $LOG
