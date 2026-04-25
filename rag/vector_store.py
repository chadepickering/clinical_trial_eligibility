"""
ChromaDB vector store operations for trial-level semantic retrieval.

Collection: oncology_trials
    - One document per trial (nct_id as ID)
    - 384-dim normalised embeddings from all-MiniLM-L6-v2
    - Metadata: nct_id, conditions, phases, status
      (enables filtered retrieval, e.g. status='RECRUITING')

Usage:
    from rag.vector_store import get_client, get_collection, upsert_trials, query_trials
"""

import chromadb
from chromadb import Collection

COLLECTION_NAME = "oncology_trials"
DEFAULT_PERSIST_DIR = "data/processed/chroma"


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client at the given directory."""
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(client: chromadb.PersistentClient) -> Collection:
    """Get or create the oncology_trials collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_trials(
    collection: Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """
    Upsert trial embeddings into the collection.

    Uses upsert (not add) so embed.py is safe to re-run — existing
    documents are overwritten rather than duplicated.

    Args:
        collection:  ChromaDB Collection object
        ids:         nct_id per trial
        embeddings:  384-dim normalised vectors
        documents:   composite text strings (stored for retrieval inspection)
        metadatas:   per-trial metadata dicts
    """
    # ChromaDB upsert accepts batches; chunk to avoid memory spikes on large runs
    chunk = 500
    for i in range(0, len(ids), chunk):
        collection.upsert(
            ids=ids[i:i + chunk],
            embeddings=embeddings[i:i + chunk],
            documents=documents[i:i + chunk],
            metadatas=metadatas[i:i + chunk],
        )


def query_trials(
    collection: Collection,
    query_embedding: list[float],
    n_results: int = 10,
    filters: dict | None = None,
    doc_max_len: int = 200,
) -> list[dict]:
    """
    Return the top-n most similar trials for a query embedding.

    Args:
        collection:      ChromaDB Collection
        query_embedding: 384-dim normalised query vector
        n_results:       number of results to return
        filters:         optional ChromaDB where-clause dict,
                         e.g. {"status": {"$eq": "RECRUITING"}}
        doc_max_len:     character limit on the returned document field.
                         200 is sufficient for display; pass 2000 when the
                         document will be fed to a cross-encoder for reranking.

    Returns:
        list of dicts with keys: nct_id, score, conditions, phases, status, document
    """
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "distances", "documents"],
    )
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    output = []
    for i, nct_id in enumerate(results["ids"][0]):
        output.append({
            "nct_id":     nct_id,
            "score":      1.0 - results["distances"][0][i],  # cosine similarity
            "conditions": results["metadatas"][0][i].get("conditions", ""),
            "phases":     results["metadatas"][0][i].get("phases", ""),
            "status":     results["metadatas"][0][i].get("status", ""),
            "document":   results["documents"][0][i][:doc_max_len],
        })

    return output


def collection_count(collection: Collection) -> int:
    """Return the number of documents in the collection."""
    return collection.count()
