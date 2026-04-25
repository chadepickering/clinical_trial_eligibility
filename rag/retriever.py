"""
Two-stage retrieval for the clinical trial RAG pipeline.

Stage 1 — Bi-encoder retrieval:
    Embed the query with all-MiniLM-L6-v2 and fetch the top-n_candidates
    from ChromaDB by cosine similarity. Fast; operates on pre-computed
    document embeddings.

Stage 2 — Cross-encoder reranking:
    Score each (query, document) pair with cross-encoder/ms-marco-MiniLM-L-6-v2
    and return the top-n_results by cross-encoder score. Slower but more
    accurate; reads both query and document jointly rather than comparing
    independent vectors.

Domain note:
    ms-marco-MiniLM-L-6-v2 was trained on MS MARCO web passage ranking.
    Clinical eligibility text is a domain shift. Whether reranking improves
    on bi-encoder precision for clinical queries is measured empirically in
    tests/test_rag.py — not assumed.

Usage:
    from rag.retriever import retrieve_and_rerank
    results = retrieve_and_rerank(query, collection, n_candidates=20, n_results=5)
"""

from sentence_transformers import CrossEncoder

from rag.embedder import embed_one
from rag.vector_store import query_trials

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def retrieve(
    query: str,
    collection,
    n_candidates: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """
    Bi-encoder retrieval: embed query and return top-n_candidates from ChromaDB.

    Documents are returned with up to 2000 characters of composite text so
    the cross-encoder reranker has sufficient context.

    Args:
        query:        natural language patient description or clinical query
        collection:   ChromaDB Collection (from rag.vector_store.get_collection)
        n_candidates: number of results to return
        filters:      optional ChromaDB where-clause dict,
                      e.g. {"status": {"$eq": "RECRUITING"}}

    Returns:
        list of trial dicts with keys: nct_id, score, conditions, phases,
        status, document (up to 2000 chars)
    """
    query_vec = embed_one(query)
    return query_trials(
        collection, query_vec,
        n_results=n_candidates,
        filters=filters,
        doc_max_len=2000,
    )


def rerank(
    query: str,
    candidates: list[dict],
    n_results: int = 5,
) -> list[dict]:
    """
    Cross-encoder reranking of bi-encoder candidates.

    Scores each (query, document) pair. Mutates each candidate dict to add
    a 'rerank_score' field, then returns the top-n_results sorted by that
    score descending.

    Args:
        query:      the original query string
        candidates: list of trial dicts from retrieve()
        n_results:  number of results to return

    Returns:
        top-n_results trial dicts, each with an added 'rerank_score' field
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["document"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:n_results]


def retrieve_and_rerank(
    query: str,
    collection,
    n_candidates: int = 20,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """
    Full two-stage retrieval: bi-encoder → cross-encoder reranking.

    Args:
        query:        natural language patient description or clinical query
        collection:   ChromaDB Collection (from rag.vector_store.get_collection)
        n_candidates: bi-encoder candidate pool size before reranking
        n_results:    final results to return after reranking
        filters:      optional ChromaDB where-clause dict

    Returns:
        list of n_results trial dicts, each with keys:
            nct_id, score (bi-encoder cosine similarity),
            rerank_score (cross-encoder logit),
            conditions, phases, status, document
    """
    candidates = retrieve(query, collection, n_candidates=n_candidates, filters=filters)
    return rerank(query, candidates, n_results=n_results)
