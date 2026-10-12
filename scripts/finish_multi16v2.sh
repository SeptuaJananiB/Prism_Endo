#!/usr/bin/env bash
# Post-training steps for DLC v2 (the original chain's controlling shell was stopped at ~01:30 by low memory).
# Waits until the training process has finished, then converts snapshots, analyses the 4 test clips with the final
# snapshot and scores on the 40 HUMAN frames.
set -e
cd "$(dirname "$0")/.."
P=$(ls -d outputs/dlc_projects/prism-tips-MULTI16v2-auto-* | tail -1)
LOG=outputs/logs/multi16v2_dlc.log
until ! powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like '*dlc_train_gpu*MULTI16v2*' }" | grep -q .; do sleep 60; done
echo "train exit (detected) $(date '+%H:%M:%S')" >> $LOG
.venv-dml/Scripts/python scripts/dlc_snapshots_to_cpu.py $P/dlc-models-pytorch >> $LOG 2>&1
.venv/Scripts/python - >> $LOG 2>&1 <<PY
import glob, deeplabcut
T = glob.glob("$P/dlc-models-pytorch/*/*/train")[0]
n = len([s for s in glob.glob(T + "/snapshot-*.pt") if "best" not in s])
vids = [f"Videos/video01_{v}.mp4" for v in (14859, 14939, 15019, 15099)]
deeplabcut.analyze_videos("$P/config.yaml", vids, shuffle=1, save_as_csv=True, destfolder="outputs/dlc/multi16v2_model",
                          snapshot_index=n - 1, device="cpu")
PY
E=$(ls -d outputs/dlc_projects/prism-tips-eval-* | tail -1)
for c in 0.5 0.1 0.0; do
  .venv/Scripts/python scripts/evaluate_dlc_heldout.py --eval-config $E/config.yaml --pred-dir outputs/dlc/multi16v2_model \
    --exclude outputs/label_qa/excluded.json --min-conf $c > outputs/logs/multi16v2_dlc_eval_c$c.log 2>&1
done
echo "all done $(date '+%H:%M:%S')" >> $LOG
