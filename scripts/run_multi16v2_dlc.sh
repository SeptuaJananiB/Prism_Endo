#!/usr/bin/env bash
# DLC v2: L-hook-balanced 16-video auto labels; continues from the MULTI16 final snapshot; scored on 40 HUMAN frames.
# Waits for the MULTI16 chain to finish (one GPU job at a time).
set -e
cd "$(dirname "$0")/.."
until grep -q "all done" outputs/logs/multi16_dlc.log 2>/dev/null; do sleep 60; done
P=$(ls -d outputs/dlc_projects/prism-tips-MULTI16v2-auto-* | tail -1)
PREV=$(ls -d outputs/dlc_projects/prism-tips-MULTI16-auto-*/dlc-models-pytorch/*/*/train | tail -1)
START=$(ls $PREV/snapshot-0*.pt | grep -v bak | sort | tail -1)
LOG=outputs/logs/multi16v2_dlc.log
echo "start $(date '+%Y-%m-%d %H:%M:%S') from $START" > $LOG
.venv-dml/Scripts/python scripts/dlc_train_gpu.py --config $P/config.yaml --epochs ${EPOCHS:-50} --batch-size 8 \
  --save-epochs 5 --snapshot "$START" >> $LOG 2>&1
echo "train exit $(date '+%H:%M:%S')" >> $LOG
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
