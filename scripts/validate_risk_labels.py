"""Validate surgeon/expert risk-event label files (and measure inter-rater agreement).

Checks (see src/risk/labels.py and docs/RISK_ANNOTATION.md): required columns,
numeric non-negative times, start_s < end_s, intervals inside the video duration
(from pipeline results.csv files and/or the video files), event types / severity /
confidence from configs/risk_events.yaml, duplicate rows, overlapping intervals of
one annotator, per-video coverage (review_complete markers). With >= 2 annotators
on the same video it reports frame-level Cohen's kappa and event-level agreement
(an event is matched if the intervals overlap) plus the unmatched events to
adjudicate. Exit code: 0 valid, 1 invalid, 2 no/unreadable label files.

  python scripts/validate_risk_labels.py outputs/annotations
  python scripts/validate_risk_labels.py outputs/annotations/risk_events_surgeonA.csv \
      outputs/annotations/risk_events_surgeonB.csv --json outputs/annotations/validation.json
  python scripts/validate_risk_labels.py labels.csv --write-merged labels_merged.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.labels import expand_label_paths, merge_overlaps, to_label_table, validate_label_files  # noqa: E402


def _json_default(o):
    try:
        return o.item()  # numpy scalars
    except AttributeError:
        return str(o)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", nargs="+", help="label CSV file(s), folders (all *.csv) or glob patterns")
    p.add_argument("--runs-dir", default="outputs/runs", help="pipeline run folders (results.csv -> durations)")
    p.add_argument("--pipeline-csv", nargs="*", default=[], help="extra pipeline results.csv files")
    p.add_argument("--videos-dir", nargs="*", default=["Videos"], help="folders with the source videos")
    p.add_argument("--schema", default=None, help="schema YAML (default configs/risk_events.yaml)")
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    p.add_argument("--json", help="write the full report (incl. per-video stats and disagreements) as JSON")
    p.add_argument("--write-merged", help="write a copy without duplicate rows and with same-type overlapping "
                                          "intervals of one annotator merged (review the result!)")
    args = p.parse_args(argv)

    paths = expand_label_paths(args.labels)
    if not paths:
        print(f"No label files found for {args.labels}.")
        return 2
    df, rep, _ = validate_label_files(paths, args.schema, args.pipeline_csv, args.runs_dir, args.videos_dir)
    print(f"files: {[str(x) for x in paths]}")
    print(rep.format())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep.as_dict(), f, indent=2, default=_json_default)
        print(f"report written to {args.json}")
    if args.write_merged:
        if df is None or rep.form != "interval":
            print("--write-merged needs readable interval-form labels.")
        else:
            ok_rows = df[df["start_s"].notna() & df["end_s"].notna() & (df["start_s"] < df["end_s"])]
            merged = merge_overlaps(ok_rows)
            to_label_table(merged).to_csv(args.write_merged, index=False)
            print(f"merged labels ({len(ok_rows)} -> {len(merged)} rows) written to {args.write_merged}; "
                  f"rows with invalid times were dropped - validate the result again.")
    if not all(x.is_file() for x in paths):
        return 2
    if not rep.ok or (args.strict and rep.warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
