#!/usr/bin/env bash
# Post-training steps for the held-out fold (video52/55 excluded).
# Waits until the training process has finished, then converts snapshots, analyses the 4 test clips with the final
# snapshot and scores on the 40 HUMAN frames.
set -e
cd "$(dirname "$0")/.."
P=$(ls -d outputs/dlc_projects/prism-tips-V3fold-holdout5255-* | tail -1)
LOG=outputs/logs/v3fold_dlc.log
until ! powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like '*dlc_train_gpu*V3fold*' }" | grep -q .; do sleep 60; done
echo "train exit (detected) $(date '+%H:%M:%S')" >> $LOG
.venv-dml/Scripts/python scripts/dlc_snapshots_to_cpu.py $P/dlc-models-pytorch >> $LOG 2>&1
# Analyse with EXACTLY the newest numbered checkpoint: copy it alone into a scratch project (DLC's own snapshot
# index also counts the "best" checkpoint, which made index -1 / n-1 pick the wrong epoch).
.venv/Scripts/python - >> $LOG 2>&1 <<PY
import glob, re, shutil, yaml, deeplabcut
from pathlib import Path
P = Path("$P"); T = Path(glob.glob(str(P / "dlc-models-pytorch/*/*/train"))[0]); M = T.parent
snaps = {int(re.search(r"(\d+)\.pt$", s.name)[1]): s for s in T.glob("snapshot-*.pt")}  # numbered or "best" (DLC may name the final epoch snapshot-best-NNN)
ep = max(snaps); Q = Path("outputs/dlc/_scratch_v3fold_project")
shutil.rmtree(Q, ignore_errors=True); Q.mkdir(parents=True)
shutil.copy2(P / "config.yaml", Q / "config.yaml"); shutil.copytree(P / "training-datasets", Q / "training-datasets")
R = M.relative_to(P); (Q / R / "train").mkdir(parents=True); shutil.copytree(M / "test", Q / R / "test")
shutil.copy2(T / "pytorch_config.yaml", Q / R / "train"); shutil.copy2(snaps[ep], Q / R / "train")
c = yaml.safe_load((Q / "config.yaml").read_text()); c["project_path"] = str(Q.resolve())
(Q / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
print("analysing with", snaps[ep].name)
vids = [f"Videos/{c}.mp4" for c in ("video52_02746", "video52_02826", "video52_02906", "video55_00508", "video55_00588", "video55_00668")]
deeplabcut.analyze_videos(str(Q / "config.yaml"), vids, shuffle=1, save_as_csv=True, destfolder="outputs/dlc/v3fold_model", device="cpu")
PY
.venv/Scripts/python scripts/eval_lhook_fold.py --dlc-dir outputs/dlc/v3fold_model > outputs/logs/v3fold_eval.log 2>&1
echo "all done $(date '+%H:%M:%S')" >> $LOG
