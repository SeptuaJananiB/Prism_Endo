# Clean (leakage-free) DLC L-hook evaluation

Date: 2026-10-11. Research prototype; pixel errors on 854x480 CholecSeg8k frames, not millimetres, not clinical accuracy.

## Why
The earlier "wider test" (`outputs/fold_eval/`, median 9.8 px) used a fold model whose stage-1 init (DLC v2) had trained on
**automatic** labels of video52/55, the two test patients. That is leakage: the test videos were seen during
pre-training, even if not with human labels. The provenance audit (`outputs/label_qa/provenance.json`) lists which
videos fed every model.

## Clean protocol (`scripts/run_clean_eval.py`)
1. **Stage 1** (`prism-tips-CLEANs1-ex5255`): DLC from ImageNet weights on automatic labels of 14 videos, **video52
   and video55 excluded** (`make_dlc_multivideo_project.py --exclude-videos`), 662 frames, 45 epochs.
2. **Stage 2** (`prism-tips-CLEANs2-ex5255`): fine-tune on human L-hook clicks from 4 videos (12, 18, 20, 37) only,
   25 epochs (lr 1e-4, then 2e-5 from epoch 65), optimizer state stripped at the stage boundary.
3. **Freeze** the last snapshot (no selection on test data) -> `outputs/models/dlc_clean_ex5255`.
4. **Test once** on the 35 human L-hook clicks of video52/55 (6 clips: video52_02746/02826/02906,
   video55_00508/00588/00668) with `scripts/eval_lhook_fold.py --out outputs/fold_eval_clean`.

No frame of video52/55 (human or automatic label) reached either training stage. Stage 1 was interrupted by a laptop
shutdown at epoch 40 and resumed from its saved snapshot (same settings) - this does not affect the test.

## Result (`outputs/fold_eval_clean/fold_results.json`)

| method | found | median px | mean px | within 20 px |
|---|---|---|---|---|
| **DLC only (clean)** | 35 / 35 | **6.5** | 11.2 | **91 %** |
| Hybrid (DLC gated by instrument mask) | 26 / 35 | 7.2 | 16.8 | 88 % |
| Geometry only | 25 / 35 | 28.6 | 60.5 | 44 % |
| previous fold model (leaky init) | 35 / 35 | 9.8 | 12.3 | 80 % |

One labelled frame had no visible L-hook; every method produced one false alarm on it.

## Interpretation
- The leakage did **not** inflate the earlier number: the clean model is at least as good (6.5 vs 9.8 px).
- The hybrid gate drops 9 tips that DLC found, because the instrument mask misses the L-hook in those frames;
  with this model, DLC alone is better. Changing the deployed default is a separate decision (not done here).
- Limits: 35 clicks, 6 clips, 2 patients, one annotator. Encouraging, not a broad validation.

## Reproduce
```
.venv-dml\Scripts\python scripts\run_clean_eval.py      # log: outputs/logs/clean_eval.log ("all done")
```
