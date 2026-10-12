#!/usr/bin/env bash
# DLC trained on automatic PSEUDO labels from 16 videos (auto_geo_v1); scored only on the 40 HUMAN eval labels.
set -e
cd "$(dirname "$0")/.."
P=$(ls -d outputs/dlc_projects/prism-tips-MULTI16-auto-* | tail -1)
LOG=outputs/logs/multi16_dlc.log
echo "start $(date '+%Y-%m-%d %H:%M:%S')" > $LOG
[ -d "$P/training-datasets" ] || .venv/Scripts/python scripts/dlc_workflow.py build --config $P/config.yaml >> $LOG 2>&1
.venv-dml/Scripts/python scripts/dlc_train_gpu.py --config $P/config.yaml --epochs ${EPOCHS:-30} --batch-size 8 --save-epochs 5 >> $LOG 2>&1
echo "train exit $(date '+%H:%M:%S')" >> $LOG
.venv-dml/Scripts/python scripts/dlc_snapshots_to_cpu.py $P/dlc-models-pytorch >> $LOG 2>&1
# analyze with the FINAL epoch snapshot (DLC would otherwise pick snapshot-best)
.venv/Scripts/python - >> $LOG 2>&1 <<PY
import glob, deeplabcut
T = glob.glob("$P/dlc-models-pytorch/*/*/train")[0]
n = len([s for s in glob.glob(T + "/snapshot-*.pt") if "best" not in s])
vids = [f"Videos/video01_{v}.mp4" for v in (14859, 14939, 15019, 15099)]
deeplabcut.analyze_videos("$P/config.yaml", vids, shuffle=1, save_as_csv=True, destfolder="outputs/dlc/multi16_model",
                          snapshot_index=n - 1, device="cpu")
PY
E=$(ls -d outputs/dlc_projects/prism-tips-eval-* | tail -1)
for c in 0.5 0.1; do
  .venv/Scripts/python scripts/evaluate_dlc_heldout.py --eval-config $E/config.yaml --pred-dir outputs/dlc/multi16_model \
    --exclude outputs/label_qa/excluded.json --min-conf $c > outputs/logs/multi16_dlc_eval_c$c.log 2>&1
done
echo "all done $(date '+%H:%M:%S')" >> $LOG
