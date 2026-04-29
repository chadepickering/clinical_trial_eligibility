"""
Sample trials from ChromaDB for evaluation case construction.

Fetches a diverse set of trials (up to --limit) and prints their
NCT ID, metadata, and full composite document to stdout. Output is
intended for manual review to construct labeled evaluation cases for
data/labeled/eval_eligible.json and data/labeled/eval_ineligible.json.

Usage:
    python scripts/sample_eval_corpus.py [--limit N] [--conditions SUBSTR]

Examples:
    # Print 200 diverse trials
    python scripts/sample_eval_corpus.py --limit 200

    # Filter to breast cancer trials only
    python scripts/sample_eval_corpus.py --conditions "Breast Cancer"
"""

import argparse
import sys

sys.path.insert(0, ".")
from rag.vector_store import get_client, get_collection, collection_count


CHROMA_DIR = "data/processed/chroma"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=200, help="Max trials to fetch (default: 200)")
    parser.add_argument("--conditions", type=str, default=None, help="Filter by conditions substring (case-insensitive)")
    args = parser.parse_args()

    client = get_client(CHROMA_DIR)
    col = get_collection(client)
    n = collection_count(col)
    print(f"# Collection size: {n:,} trials\n", file=sys.stderr)

    result = col.get(limit=args.limit, include=["documents", "metadatas"])

    printed = 0
    for nct_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        conditions = meta.get("conditions", "")
        if args.conditions and args.conditions.lower() not in conditions.lower():
            continue

        print(f"=== {nct_id} ===")
        print(f"SEX: {meta.get('sex','?')} | MIN_AGE: {meta.get('min_age','?')} | MAX_AGE: {meta.get('max_age','?')}")
        print(f"STATUS: {meta.get('status','?')} | PHASES: {meta.get('phases','?')}")
        print(f"CONDITIONS: {conditions}")
        print()
        print(doc[:3000])
        if len(doc) > 3000:
            print(f"[... document truncated at 3000/{len(doc)} chars ...]")
        print()
        printed += 1

    print(f"# Printed {printed} trials", file=sys.stderr)


if __name__ == "__main__":
    main()
