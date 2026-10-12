# Surgeon risk / near-miss event annotation

No risk labels exist. Until a surgeon creates them, PRISM-Endo uses only the **rule-based research
score**, which is uncalibrated and is not a probability of injury.

## Schema

`configs/risk_events.yaml` (status: **proposal**; the event definitions must be confirmed by the
supervising surgeon first). One CSV per annotator:

```
video_id,start_s,end_s,label,event_type,severity,annotator,confidence,notes,blind
video01_14859,1.20,1.84,1,near_miss_critical_structure,2,surgeonA,3,"hook within ~2 mm of liver bed",true
```

* `video_id` = the pipeline's `video_id` (input file / folder name). Times are seconds (`timestamp_s`).
* Annotate **whole videos**: everything outside an interval of an annotated video counts as negative.
* Work **blind** to the PRISM overlay and score (annotate from `Videos/*.mp4` or `original_h264.mp4`).
* At least two independent annotators on a shared subset, then adjudication; report agreement.

## Validate, then train

```powershell
.venv\Scripts\python scripts\validate_risk_labels.py outputs\annotations\risk_events_surgeonA.csv outputs\annotations\risk_events_surgeonB.csv --runs-dir outputs\runs
.venv\Scripts\python scripts\train_risk_model.py --pipeline-csv outputs\runs\<video>\results.csv ... --labels <merged.csv> --test-videos <ids>
```

The validator checks columns, times, overlaps, event types and coverage, and computes inter-rater
agreement (frame-level Cohen's kappa, event matching). Training refuses when labels are missing or
invalid, when there are fewer than 2 videos, or when there are too few events. It uses video-level
train/test splits and reports precision, recall, F1, AUROC, AUPRC and prevalence.

## Annotation tool

```powershell
.venv\Scripts\streamlit run src\ui\annotate_events.py --server.port 8504
```

Enter a pseudonymous annotator ID, pick a video (the original, without the PRISM overlay), scrub with the
frame slider, set START/END, then choose the type, severity and confidence and save. Rows go to
`outputs/annotations/risk_events_<annotator>.csv`. When a video is finished, press
**Mark whole video reviewed** so the rest of that video counts as negative.
