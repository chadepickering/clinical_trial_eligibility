"""
End-to-end RAG pipeline — retrieval → reranking → generation.

Orchestrates the three upstream modules into a single callable:

    1. retrieve_and_rerank (rag/retriever.py)
       Bi-encoder retrieval from ChromaDB, cross-encoder reranking.

    2. assess_trial (rag/generator.py)
       Per-trial eligibility Q&A via Mistral-7B through Ollama.
       One generate() call per reranked trial.

Design: custom thin pipeline (Option B from the Step 8 outline).
LlamaIndex is not used. All retrieval logic lives in the already-tested
rag/vector_store.py. The pipeline is ~30 lines of orchestration code.

Architecture note:
    The LLM verdict (ELIGIBLE / NOT ELIGIBLE / UNCERTAIN) returned here is
    a narrative output for the clinician — it has no formal weight in the
    Bayesian posterior. The Bayesian scorer operates independently
    on SciBERT labels and NER-extracted thresholds. See PIPELINE_WALKTHROUGH.md
    for the full explanation of these non-overlapping roles.

Usage:
    from rag.pipeline import run_pipeline
    from rag.vector_store import get_client, get_collection

    collection = get_collection(get_client("data/processed/chroma"))
    results = run_pipeline(
        query="Female, 52yo, BRCA1 mutation, platinum-sensitive recurrent OC",
        collection=collection,
    )
    for r in results:
        print(r["nct_id"], r["verdict"], r["score"])
"""

import time

from rag.generator import assess_trial
from rag.retriever import retrieve_and_rerank

DEFAULT_N_CANDIDATES = 20   # bi-encoder candidate pool
DEFAULT_N_RESULTS    = 5    # trials returned after reranking + generation


def run_pipeline(
    query: str,
    collection,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    n_results: int = DEFAULT_N_RESULTS,
    filters: dict | None = None,
    model: str = "mistral",
    generate: bool = True,
) -> list[dict]:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    Args:
        query:        free-text patient description or clinical query
        collection:   ChromaDB Collection (from rag.vector_store.get_collection)
        n_candidates: bi-encoder candidate pool size before reranking
        n_results:    number of trials to return after reranking
        filters:      optional ChromaDB where-clause dict,
                      e.g. {"status": {"$eq": "RECRUITING"}}
        model:        Ollama model name (must be pulled)
        generate:     if False, skip LLM generation and return retrieval
                      results only. Useful for latency profiling and tests
                      that do not require Ollama.

    Returns:
        list of n_results dicts, one per trial, with keys:
            nct_id        — trial identifier
            score         — bi-encoder cosine similarity [0, 1]
            rerank_score  — cross-encoder logit
            conditions    — trial condition string from ChromaDB metadata
            phases        — trial phase string from ChromaDB metadata
            status        — trial status string from ChromaDB metadata
            document      — composite trial text (truncated to 2000 chars)
            verdict       — "ELIGIBLE", "NOT ELIGIBLE", "UNCERTAIN", or None
                            (None when generate=False)
            explanation   — LLM-generated prose assessment, or None
            latency_s     — wall-clock seconds for this trial's generate() call,
                            or None when generate=False
    """
    # Stage 1 — bi-encoder retrieval + cross-encoder reranking
    reranked = retrieve_and_rerank(
        query,
        collection,
        n_candidates=n_candidates,
        n_results=n_results,
        filters=filters,
    )

    # Stage 2 — per-trial LLM assessment
    results = []
    for trial in reranked:
        if generate:
            t0 = time.time()
            assessment = assess_trial(
                nct_id=trial["nct_id"],
                trial_document=trial["document"],
                patient_query=query,
                model=model,
            )
            latency = round(time.time() - t0, 2)
            verdict     = assessment["verdict"]
            explanation = assessment["explanation"]
        else:
            latency     = None
            verdict     = None
            explanation = None

        results.append({
            **trial,
            "verdict":     verdict,
            "explanation": explanation,
            "latency_s":   latency,
        })

    return results
