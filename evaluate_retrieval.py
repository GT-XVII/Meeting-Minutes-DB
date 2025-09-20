
#!/usr/bin/env python3

"""
evaluate_retrieval.py

Compute Recall@k, nDCG@k, and MRR for vector database results.
- Works with a simple JSON input format (see Example section below).
- Prints per-query metrics and overall averages.
- Optionally pretty-prints each query's retrieved docs in a style similar to the provided sample.

No external packages required.

----------
Input format
----------
The script expects a JSON file containing a list of query records. Each record has:
{
  "query": "string",
  "k": 5,                            # optional; overrides --k for this query
  "elapsed_sec": 0.021,              # optional; time to retrieve (seconds)
  "retrieved": [                     # ranked list in returned order (best first)
    {
      "doc_id": "string",            # required (unique id for doc)
      "score": 0.42,                 # optional
      "metadata": {                  # optional (any keys)
        "source": "Daily Lot 1&2 060825.txt",
        "date_str": "06/08/2025",
        "Meeting_type": "Daily Lot 1&2",
        "date": 1754438400.0,
        "_id": "3df559cb16d447e9afcbd9054ddff371",
        "_collection_name": "document_vectors"
      }
    },
    ...
  ],
  # Relevance can be provided in either of two ways:
  # 1) Binary relevant doc ids:
  "relevant_ids": ["3df559...", "e63153..."]
  # 2) Graded qrels mapping doc_id -> nonnegative relevance grade (0,1,2,...):
  # "qrels": {"3df559...": 2, "e63153...": 1}
}

----------
Example
----------
python evaluate_retrieval.py --data example_results.json --k 5 --pretty
python evaluate_retrieval.py --data example_results.json --k 10 --csv metrics.csv
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from typing import Dict, Iterable, List, Optional, Tuple


def recall_at_k(rels: Dict[str, float], ranking: List[str], k: int) -> float:
    """
    Binary Recall@k. Treats any positive grade (>0) as relevant.
    rels: dict of doc_id -> grade (0 for non-relevant, >0 for relevant)
    ranking: list of doc_ids in ranked order (best first)
    """
    if not rels:
        return 0.0
    relevant_ids = {d for d, g in rels.items() if g > 0}
    if not relevant_ids:
        return 0.0
    topk = set(ranking[:k])
    hits = len(relevant_ids & topk)
    return hits / float(len(relevant_ids))


def dcg_at_k(gains: Iterable[float], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        # Standard log2 discount; position 1 has log2(2)=1
        denom = math.log2(i + 1)
        dcg += (g / denom)
    return dcg


def ndcg_at_k(rels: Dict[str, float], ranking: List[str], k: int) -> float:
    """
    nDCG@k with graded relevance if provided, else binary.
    rels: dict of doc_id -> grade (0,1,2,...). Missing doc_ids implicitly 0.
    ranking: list of doc_ids in ranked order.
    """
    gains = [rels.get(doc_id, 0.0) for doc_id in ranking]
    dcg = dcg_at_k(gains, k)
    # Ideal DCG: sort by grade descending
    ideal_gains = sorted(rels.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr(rels: Dict[str, float], ranking: List[str]) -> float:
    """
    Mean Reciprocal Rank for first relevant (grade>0) item.
    """
    for idx, doc_id in enumerate(ranking, start=1):
        if rels.get(doc_id, 0.0) > 0:
            return 1.0 / float(idx)
    return 0.0


def to_qrels(relevant_ids: Optional[List[str]] = None,
             qrels: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Normalize relevance to a dict of doc_id -> grade.
    If both are given, 'qrels' takes precedence.
    """
    if qrels is not None:
        # Coerce to float for safety
        return {str(k): float(v) for k, v in qrels.items()}
    if relevant_ids is not None:
        return {str(doc_id): 1.0 for doc_id in relevant_ids}
    return {}  # no relevance available


def pretty_print_query(record: dict, k: int) -> None:
    query = record.get("query", "<no query>")
    elapsed = record.get("elapsed_sec", None)
    retrieved = record.get("retrieved", [])
    print(f"Query: {query}")
    if elapsed is not None:
        print(f"Retrieved {min(len(retrieved), k)} docs in {elapsed:.3f}s\n")
    else:
        print(f"Retrieved {min(len(retrieved), k)} docs\n")

    for i, item in enumerate(retrieved[:k], start=1):
        md = item.get("metadata", {})
        print(f"[Doc {i} Metadata]")
        if md:
            # Print stable keys first, then the rest
            stable_keys = ["source", "date_str", "Meeting_type", "date", "_id", "_collection_name"]
            for key in stable_keys:
                if key in md:
                    print(f"{key}: {md[key]}")
            # Print any other metadata keys
            for key, val in md.items():
                if key not in stable_keys:
                    print(f"{key}: {val}")
        else:
            print(f"doc_id: {item.get('doc_id')}")
            if "score" in item:
                print(f"score: {item['score']}")
        print("")


def evaluate_record(record: dict, default_k: int) -> Tuple[float, float, float]:
    ranking_ids = [str(item.get("doc_id")) for item in record.get("retrieved", [])]
    k = int(record.get("k", default_k))
    rels = to_qrels(record.get("relevant_ids"), record.get("qrels"))

    rec = recall_at_k(rels, ranking_ids, k)
    nd = ndcg_at_k(rels, ranking_ids, k)
    rr = mrr(rels, ranking_ids)
    return rec, nd, rr


def main():
    parser = argparse.ArgumentParser(description="Evaluate vector DB results with Recall@k, nDCG@k, and MRR.")
    parser.add_argument("--data", required=True, help="Path to JSON file (list of query records).")
    parser.add_argument("--k", type=int, default=5, help="Cutoff k for Recall@k and nDCG@k (can be overridden per-record).")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print retrieved docs for each query.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write per-query metrics as CSV.")
    args = parser.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read JSON data file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(records, list):
        print("ERROR: Data file must contain a JSON list of query records.", file=sys.stderr)
        sys.exit(1)

    rows = []
    rec_list, nd_list, rr_list = [], [], []

    for idx, rec in enumerate(records, start=1):
        k = int(rec.get("k", args.k))
        if args.pretty:
            pretty_print_query(rec, k)

        r, n, rr = evaluate_record(rec, args.k)
        rec_list.append(r)
        nd_list.append(n)
        rr_list.append(rr)
        query = rec.get("query", f"q{idx}")
        elapsed = rec.get("elapsed_sec", None)
        rows.append({
            "query": query,
            "k": k,
            "elapsed_sec": elapsed,
            "recall_at_k": round(r, 4),
            "ndcg_at_k": round(n, 4),
            "mrr": round(rr, 4),
        })

    # Print a compact summary table
    print("Per-query metrics")
    print("-----------------")
    header = ["query", "k", "elapsed_sec", "recall_at_k", "ndcg_at_k", "mrr"]
    print(",".join(header))
    for row in rows:
        print(",".join(str(row[h]) if row[h] is not None else "" for h in header))

    # Averages
    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    avg_recall = avg(rec_list)
    avg_ndcg = avg(nd_list)
    avg_mrr = avg(rr_list)

    print("\nAverages")
    print("--------")
    print(f"Recall@k: {avg_recall:.4f}")
    print(f"nDCG@k:   {avg_ndcg:.4f}")
    print(f"MRR:      {avg_mrr:.4f}")

    # Optional CSV
    if args.csv:
        try:
            import csv
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"\nWrote per-query metrics to: {args.csv}")
        except Exception as e:
            print(f"\nERROR: Could not write CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
