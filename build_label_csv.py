
#!/usr/bin/env python3

"""
build_label_csv.py

Create a labeling CSV from your retrieval dump (the same JSON used by evaluate_retrieval.py).
You only label the *retrieved* docs (no need to look at the whole corpus).

Usage:
  python build_label_csv.py --data results.json --out labels.csv --pool-k 20

This writes labels.csv with columns:
  query_id,query,doc_rank,doc_id,source,date_str,Meeting_type,preview,label

- preview will try: item["text"] or item["content"] or item["chunk"], if present.
- label: fill 0 (not relevant), 1 (relevant), or any integer grade (2,3,...) if you want graded relevance.
"""

import argparse
import json
import csv
import sys


def pick_preview(item: dict) -> str:
    # Try to get a short text preview from common keys if available
    for key in ("text", "content", "chunk", "snippet"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v[:500]
    md = item.get("metadata", {})
    # fallback: join a few useful metadata bits for context
    bits = []
    for k in ("source", "date_str", "Meeting_type"):
        if k in md:
            bits.append(f"{k}={md[k]}")
    return " | ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to retrieval results JSON")
    ap.add_argument("--out", required=True, help="Path to write label CSV")
    ap.add_argument("--pool-k", type=int, default=20, help="How many top results per query to label")
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

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id","query","doc_rank","doc_id","source","date_str","Meeting_type","preview","label"])
        for qi, rec in enumerate(records, start=1):
            query = rec.get("query", f"q{qi}")
            retrieved = rec.get("retrieved", [])
            for rank, item in enumerate(retrieved[:args.pool_k], start=1):
                md = item.get("metadata", {})
                writer.writerow([
                    qi,
                    query,
                    rank,
                    item.get("doc_id",""),
                    md.get("source",""),
                    md.get("date_str",""),
                    md.get("Meeting_type",""),
                    pick_preview(item),
                    ""  # label to be filled by human (0/1/2/...)
                ])

    print(f"Wrote labeling template: {args.out}")


if __name__ == "__main__":
    main()
