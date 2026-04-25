"""
RAG pipeline tests (step 8).

Two test classes:

  TestRetrieverContract — structural correctness; no ChromaDB dependency.
                          Always runnable.
  TestRetrieverQuality  — live ChromaDB queries; cross-encoder benchmark.
                          Require the collection to be populated (run embed.py
                          first). Skip guard matches test_embed.py convention.

Run all:
    PYTHONPATH=. pytest tests/test_rag.py -v

Skip quality tests if collection not built:
    PYTHONPATH=. pytest tests/test_rag.py -v -k "not Quality"
"""

import pytest

from rag.retriever import retrieve, rerank, retrieve_and_rerank
from rag.vector_store import get_client, get_collection, collection_count

CHROMA_DIR = "data/processed/chroma"

# Query used for the 8b smoke test and all quality benchmarks.
BRCA1_QUERY = (
    "Female patient, 52 years old, BRCA1 mutation, platinum-sensitive "
    "recurrent ovarian cancer, prior bevacizumab"
)

# NCT ID confirmed to contain both 'BRCA Status' and 'Ovarian Cancer' in
# conditions — used as a precision anchor, not as a rank pin.
BRCA_OVARIAN_NCT = "NCT02222883"

# Keywords indicating a result is clinically relevant for the BRCA1 query.
RELEVANT_KEYWORDS = {
    "ovarian", "fallopian", "peritoneal", "gynecol",
    "uterine", "endometrial", "cervical", "vulvar",
    "brca", "hereditary breast",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _precision(results: list[dict], keywords: set) -> int:
    """Count results whose conditions contain at least one keyword."""
    count = 0
    for r in results:
        cond_lower = r["conditions"].lower()
        if any(kw in cond_lower for kw in keywords):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def collection():
    client = get_client(CHROMA_DIR)
    col = get_collection(client)
    if collection_count(col) == 0:
        pytest.skip("ChromaDB collection is empty — run embed.py first")
    return col


# ---------------------------------------------------------------------------
# C — Retriever contract (no ChromaDB)
# ---------------------------------------------------------------------------

class TestRetrieverContract:

    def test_c1_rerank_empty_candidates_returns_empty(self):
        """rerank() on an empty candidate list must return an empty list."""
        result = rerank(BRCA1_QUERY, [])
        assert result == []

    def test_c2_rerank_fewer_candidates_than_n_results(self):
        """
        When fewer candidates are supplied than n_results, rerank() returns
        all of them rather than raising an error or padding with None.
        """
        fake_candidates = [
            {
                "nct_id": "NCT000001",
                "score": 0.8,
                "conditions": "Ovarian Cancer",
                "phases": "PHASE2",
                "status": "RECRUITING",
                "document": "Ovarian cancer trial with bevacizumab and BRCA1 mutation eligibility.",
            }
        ]
        results = rerank(BRCA1_QUERY, fake_candidates, n_results=5)
        assert len(results) == 1, (
            f"Expected 1 result (all candidates), got {len(results)}"
        )

    def test_c3_rerank_adds_rerank_score_field(self):
        """
        rerank() must add a 'rerank_score' key to every returned dict.
        Uses a minimal real-text candidate so the cross-encoder has something
        to score.
        """
        fake_candidates = [
            {
                "nct_id": "NCT000001",
                "score": 0.75,
                "conditions": "Ovarian Cancer",
                "phases": "PHASE3",
                "status": "RECRUITING",
                "document": (
                    "Phase 3 trial for platinum-sensitive recurrent ovarian cancer. "
                    "BRCA1 or BRCA2 mutation required. Prior bevacizumab allowed."
                ),
            },
            {
                "nct_id": "NCT000002",
                "score": 0.60,
                "conditions": "Breast Cancer",
                "phases": "PHASE2",
                "status": "COMPLETED",
                "document": (
                    "Phase 2 trial for HER2-positive breast cancer. "
                    "No prior trastuzumab. ECOG 0-1."
                ),
            },
        ]
        results = rerank(BRCA1_QUERY, fake_candidates, n_results=2)
        assert len(results) == 2
        for r in results:
            assert "rerank_score" in r, (
                f"Missing 'rerank_score' key in result for {r['nct_id']}"
            )
            assert isinstance(r["rerank_score"], float)

    def test_c4_rerank_respects_n_results_cap(self):
        """rerank() must return at most n_results items."""
        fake_candidates = [
            {
                "nct_id": f"NCT{i:06d}",
                "score": 0.9 - i * 0.05,
                "conditions": "Ovarian Cancer",
                "phases": "PHASE2",
                "status": "RECRUITING",
                "document": f"Ovarian cancer eligibility trial document number {i}.",
            }
            for i in range(10)
        ]
        results = rerank(BRCA1_QUERY, fake_candidates, n_results=3)
        assert len(results) <= 3, (
            f"Expected ≤3 results, got {len(results)}"
        )


# ---------------------------------------------------------------------------
# D — Retriever quality (requires populated ChromaDB)
# ---------------------------------------------------------------------------

class TestRetrieverQuality:

    def test_d1_smoke_retrieve_and_rerank_returns_expected_fields(self, collection):
        """
        8b smoke test. retrieve_and_rerank() with the BRCA1 ovarian query
        must return 5 results each containing the required fields.
        This is the query run interactively during 8b verification.
        """
        results = retrieve_and_rerank(
            BRCA1_QUERY, collection, n_candidates=20, n_results=5
        )
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        required_keys = {"nct_id", "score", "rerank_score", "conditions",
                         "phases", "status", "document"}
        for r in results:
            missing = required_keys - r.keys()
            assert not missing, (
                f"Result for {r.get('nct_id', '?')} missing keys: {missing}"
            )

    def test_d2_top_biencoder_result_is_relevant_and_high_scoring(self, collection):
        """
        The top bi-encoder result for the BRCA1 query should be a gynecologic
        or BRCA-related trial with a cosine similarity > 0.65.

        Observed in 8b: NCT02222883 (BRCA Status + Ovarian Cancer, score 0.713)
        ranks first by both bi-encoder and reranker. Pinning the specific NCT ID
        would be brittle; instead we assert the conditions are relevant and the
        score reflects a strong semantic match.
        """
        candidates = retrieve(BRCA1_QUERY, collection, n_candidates=5)
        top = max(candidates, key=lambda r: r["score"])
        cond_lower = top["conditions"].lower()
        matched = [kw for kw in RELEVANT_KEYWORDS if kw in cond_lower]
        assert matched, (
            f"Top bi-encoder result {top['nct_id']} is not gynecologic/BRCA-related "
            f"(conditions: {top['conditions'][:80]}, score: {top['score']:.3f})"
        )
        assert top["score"] > 0.65, (
            f"Top bi-encoder score {top['score']:.3f} below 0.65 — "
            "embedding may have regressed or collection changed"
        )

    def test_d3_reranking_changes_result_order(self, collection):
        """
        The cross-encoder must produce a different ranking from the bi-encoder
        for the BRCA1 query. If the reranker is broken or producing constant
        scores, the order would be identical to the bi-encoder top-5.

        Observed in 8b: reranker drops breast cancer trials (bi ranks 3–4) and
        promotes lower-ranked ovarian trials — the order changes substantially.
        """
        candidates = retrieve(BRCA1_QUERY, collection, n_candidates=20)
        biencoder_ids = [r["nct_id"] for r in
                         sorted(candidates, key=lambda x: x["score"], reverse=True)[:5]]

        reranked = rerank(BRCA1_QUERY, candidates, n_results=5)
        reranked_ids = [r["nct_id"] for r in reranked]

        assert biencoder_ids != reranked_ids, (
            "Reranked order is identical to bi-encoder order — "
            "cross-encoder may not be scoring correctly"
        )

    def test_d4_biencoder_precision_baseline(self, collection):
        """
        Bi-encoder precision@5 baseline for the BRCA1 ovarian query.

        Observed in 8b: 4 of the top-5 bi-encoder results are gynecologic or
        BRCA-related oncology trials. This test pins that baseline so any future
        regression in the embedding model or collection is caught.
        """
        candidates = retrieve(BRCA1_QUERY, collection, n_candidates=5)
        p = _precision(candidates, RELEVANT_KEYWORDS)
        assert p >= 4, (
            f"Bi-encoder precision@5 dropped to {p}/5 for BRCA1 query. "
            f"Expected ≥4. Results: "
            f"{[(r['nct_id'], r['conditions'][:50]) for r in candidates]}"
        )

    def test_d5_reranking_does_not_degrade_precision(self, collection):
        """
        Cross-encoder benchmark: reranked precision@5 must be ≥ bi-encoder
        precision@5 − 1 on the BRCA1 query.

        This is NOT an assertion that reranking improves results — it is an
        assertion that it does not significantly degrade them. A degradation of
        2+ trials (e.g. bi-encoder 4/5, reranked 2/5) would indicate the
        cross-encoder domain shift is unacceptably harmful and warrants
        dropping reranking from the pipeline.

        Observed in 8b: both are 4/5. The reranker removes breast cancer trials
        (positive) but promotes one lung cancer trial (negative). Net: same
        precision, different composition.
        """
        candidates = retrieve(BRCA1_QUERY, collection, n_candidates=20)
        biencoder_top5 = sorted(candidates, key=lambda x: x["score"], reverse=True)[:5]
        reranked_top5 = rerank(BRCA1_QUERY, [c.copy() for c in candidates], n_results=5)

        biencoder_p = _precision(biencoder_top5, RELEVANT_KEYWORDS)
        reranked_p = _precision(reranked_top5, RELEVANT_KEYWORDS)

        assert reranked_p >= biencoder_p - 1, (
            f"Reranking significantly degraded precision: "
            f"bi-encoder {biencoder_p}/5 → reranked {reranked_p}/5. "
            f"Reranked results: "
            f"{[(r['nct_id'], r['conditions'][:50]) for r in reranked_top5]}"
        )
