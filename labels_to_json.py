
#!/usr/bin/env python3

"""
labels_to_json.py

Merge human labels from a CSV back into the retrieval JSON as graded qrels.

Usage:
  python labels_to_json.py --data results.json --labels labels.csv --out results_with_qrels.json

labels.csv must have columns:
  query_id,query,doc_rank,doc_id,source,date_str,Meeting_type,preview,label

Notes:
- label values are parsed as integers; non-numeric or blank labels are treated as missing and ignored.
- The script attaches a "qrels" dict per query record: {doc_id: grade}
- Existing "relevant_ids" or "qrels" in the JSON will be overwritten for those doc_ids present in the CSV.
"""

import argparse
import json
import csv
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to retrieval results JSON")
    ap.add_argument("--labels", required=True, help="Path to label CSV")
    ap.add_argument("--out", required=True, help="Path to write merged JSON")
    args = ap.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        print(f"ERROR reading {args.data}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(records, list):
        print("ERROR: JSON root must be a list of query records.", file=sys.stderr)
        sys.exit(1)

    # Load labels, grouped by query_id
    labels_by_qid = defaultdict(dict)  # qid -> {doc_id: grade}
    try:
        with open(args.labels, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    qid = int(row.get("query_id","").strip() or "0")
                except:
                    continue
                doc_id = (row.get("doc_id") or "").strip()
                label_raw = (row.get("label") or "").strip()
                if not doc_id or not label_raw:
                    continue
                try:
                    grade = int(label_raw)
                except ValueError:
                    # ignore non-numeric labels
                    continue
                if grade < 0:
                    grade = 0
                labels_by_qid[qid][doc_id] = float(grade)
    except Exception as e:
        print(f"ERROR reading {args.labels}: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge into JSON
    for qi, rec in enumerate(records, start=1):
        qrels = labels_by_qid.get(qi, {})
        if not qrels:
            continue
        rec["qrels"] = qrels  # overwrite / attach graded labels
        if "relevant_ids" in rec:
            del rec["relevant_ids"]  # avoid confusion

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote merged JSON with qrels: {args.out}")
    except Exception as e:
        print(f"ERROR writing {args.out}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
