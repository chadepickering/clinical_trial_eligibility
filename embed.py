"""
Embedding pipeline CLI — reads trials from DuckDB, embeds, writes to ChromaDB.

Usage:
    python embed.py                        # embed all trials not yet in ChromaDB
    python embed.py --reprocess            # re-embed all trials (upsert overwrites)
    python embed.py --spot-check           # query with a sample patient profile and exit
    python embed.py --db data/processed/trials.duckdb
    python embed.py --chroma data/processed/chroma
"""

import argparse
import time

import duckdb

from rag.embedder import build_corpus, embed_corpus, _get_model
from rag.vector_store import (
    get_client,
    get_collection,
    upsert_trials,
    query_trials,
    collection_count,
)

SPOT_CHECK_QUERY = (
    "Female patient, 52 years old, BRCA1 mutation, platinum-sensitive "
    "recurrent ovarian cancer, prior bevacizumab"
)


def load_trials(db_path: str, reprocess: bool, existing_ids: set[str]) -> list[dict]:
    """Load trials from DuckDB, skipping already-embedded IDs unless reprocess."""
    conn = duckdb.connect(db_path, read_only=True)
    rows = conn.execute("""
        SELECT nct_id, brief_title, brief_summary, eligibility_text,
               conditions, phases, status
        FROM trials
        WHERE eligibility_text IS NOT NULL
          AND length(eligibility_text) > 0
    """).fetchall()
    conn.close()

    cols = ['nct_id', 'brief_title', 'brief_summary', 'eligibility_text',
            'conditions', 'phases', 'status']
    all_trials = [dict(zip(cols, row)) for row in rows]

    if reprocess:
        return all_trials

    return [t for t in all_trials if t['nct_id'] not in existing_ids]


def run_embed(db_path: str, chroma_dir: str, reprocess: bool) -> None:
    client = get_client(chroma_dir)
    collection = get_collection(client)

    existing = set(collection.get(include=[])['ids']) if not reprocess else set()
    print(f"ChromaDB: {len(existing):,} trials already embedded", flush=True)

    trials = load_trials(db_path, reprocess, existing)
    print(f"Trials to embed: {len(trials):,}", flush=True)

    if not trials:
        print("Nothing to do.")
        return

    texts, metadatas, ids = build_corpus(trials)

    print("Encoding embeddings...", flush=True)
    t0 = time.time()
    embeddings = embed_corpus(texts, batch_size=64, show_progress=True)
    elapsed = time.time() - t0
    print(f"Encoded {len(embeddings):,} embeddings in {elapsed:.0f}s "
          f"({len(embeddings)/elapsed:.0f} docs/s)", flush=True)

    print("Writing to ChromaDB...", flush=True)
    upsert_trials(collection, ids, embeddings, texts, metadatas)

    total = collection_count(collection)
    print(f"\nDone. ChromaDB collection now contains {total:,} trials.", flush=True)


def run_spot_check(chroma_dir: str) -> None:
    client = get_client(chroma_dir)
    collection = get_collection(client)

    n = collection_count(collection)
    if n == 0:
        print("Collection is empty. Run embed.py first.")
        return

    print(f"Collection: {n:,} trials")
    print(f"\nQuery: {SPOT_CHECK_QUERY}\n")

    model = _get_model()
    query_vec = model.encode(
        SPOT_CHECK_QUERY, normalize_embeddings=True
    ).tolist()

    results = query_trials(collection, query_vec, n_results=10)
    print(f"{'Rank':<5} {'Score':>6}  {'NCT ID':<14}  {'Status':<12}  {'Conditions'}")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        cond = r['conditions'][:50]
        print(f"{rank:<5} {r['score']:>6.3f}  {r['nct_id']:<14}  {r['status']:<12}  {cond}")


def main():
    parser = argparse.ArgumentParser(description="Embed trials into ChromaDB")
    parser.add_argument('--db',          default='data/processed/trials.duckdb')
    parser.add_argument('--chroma',      default='data/processed/chroma')
    parser.add_argument('--reprocess',   action='store_true',
                        help='re-embed all trials (upsert overwrites existing)')
    parser.add_argument('--spot-check',  action='store_true',
                        help='query with sample patient profile and exit')
    args = parser.parse_args()

    if args.spot_check:
        run_spot_check(args.chroma)
        return

    run_embed(args.db, args.chroma, args.reprocess)


if __name__ == '__main__':
    main()
