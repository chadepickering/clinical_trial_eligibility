"""
Embedding pipeline (step 7) tests.

Two test classes:

  TestEmbeddingGeometry  — model sanity, no ChromaDB dependency
  TestRetrieval          — live ChromaDB queries; require the collection to be
                           populated (run embed.py first)

Run all:
    PYTHONPATH=. pytest tests/test_embed.py -v

Skip retrieval tests if collection not built yet:
    PYTHONPATH=. pytest tests/test_embed.py -v -k "not Retrieval"
"""

import math

import pytest

from rag.embedder import embed_one, _get_model
from rag.vector_store import (
    get_client,
    get_collection,
    query_trials,
    collection_count,
)

CHROMA_DIR = "data/processed/chroma"
WALKTHROUGH_NCT_ID = "NCT00127920"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _embed_one(text: str) -> list[float]:
    return embed_one(text)


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


@pytest.fixture(scope="module")
def model():
    return _get_model()


# ---------------------------------------------------------------------------
# A — Embedding geometry
# ---------------------------------------------------------------------------

class TestEmbeddingGeometry:

    def test_a1_normalization(self):
        """Embeddings must have unit norm (normalize_embeddings=True)."""
        vec = _embed_one("Patients with advanced ovarian carcinoma.")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}"

    def test_a2_dimension(self):
        """all-MiniLM-L6-v2 produces 384-dimensional embeddings."""
        vec = _embed_one("ECOG performance status ≤ 2.")
        assert len(vec) == 384, f"Expected 384 dims, got {len(vec)}"

    def test_a3_semantic_ordering(self):
        """Synonymous clinical terms should be closer than unrelated terms."""
        anchor = _embed_one("ovarian cancer")
        synonym = _embed_one("ovarian neoplasm")
        unrelated = _embed_one("prostate cancer")
        sim_syn = _cosine_sim(anchor, synonym)
        sim_unrel = _cosine_sim(anchor, unrelated)
        assert sim_syn > sim_unrel, (
            f"Expected sim(ovarian cancer, ovarian neoplasm)={sim_syn:.3f} "
            f"> sim(ovarian cancer, prostate cancer)={sim_unrel:.3f}"
        )

    def test_a4_determinism(self):
        """Same input must produce identical embedding on two calls."""
        text = "Creatinine ≤ 1.5 mg/dL and adequate hepatic function."
        vec1 = _embed_one(text)
        vec2 = _embed_one(text)
        assert vec1 == vec2, "Embeddings are non-deterministic"

    def test_a5_out_of_domain_low_similarity(self):
        """Clinical eligibility text should not be close to unrelated domain text."""
        clinical = _embed_one(
            "Histologically confirmed stage III ovarian carcinoma eligibility criteria"
        )
        sports = _embed_one("baseball batting average statistics home run season")
        sim = _cosine_sim(clinical, sports)
        assert sim < 0.3, f"Expected sim < 0.3 for out-of-domain text, got {sim:.3f}"


# ---------------------------------------------------------------------------
# B — Retrieval correctness (requires populated ChromaDB)
# ---------------------------------------------------------------------------

class TestRetrieval:

    def test_b1_disease_area_precision(self, collection):
        """
        A treatment-naive advanced ovarian cancer query should return only
        gynecologic oncology trials in the top 10.

        NCT00127920 is a generic first-line ovarian cancer trial whose
        eligibility text is nearly identical to dozens of similar trials in the
        corpus — pinning it to a specific rank is not a meaningful test.
        What matters is that the top 10 are all semantically correct results
        (ovarian, fallopian tube, peritoneal, or related gynecologic cancers).
        """
        query = (
            "Newly diagnosed female patient with stage III or IV ovarian carcinoma "
            "or peritoneal cancer, no prior chemotherapy or radiotherapy"
        )
        gynecologic_keywords = {
            "ovarian", "fallopian", "peritoneal", "gynecol", "uterine",
            "endometrial", "cervical", "vulvar",
        }
        vec = _embed_one(query)
        results = query_trials(collection, vec, n_results=10)
        assert len(results) == 10, "Expected 10 results"
        for r in results:
            cond_lower = r["conditions"].lower()
            matched = [kw for kw in gynecologic_keywords if kw in cond_lower]
            assert matched, (
                f"Off-target trial in top 10: {r['nct_id']} "
                f"(score={r['score']:.3f}, conditions={r['conditions'][:80]})"
            )

    def test_b2_self_retrieval(self, collection):
        """
        The walkthrough trial's own brief_title should retrieve it in the top 10.

        With mean-pooled chunk embeddings the document vector is a weighted
        average over all content (title + summary + eligibility text). A short
        title-only query matches only the first chunk's signal, so the
        similarity is diluted relative to single-pass encoding. Rank within
        top-10 in a corpus of 15,010 trials remains a strong result.
        """
        query = (
            "Pilot Study of Taxol, Carboplatin, and Bevacizumab in "
            "Advanced Stage Ovarian Carcinoma Patients"
        )
        vec = _embed_one(query)
        results = query_trials(collection, vec, n_results=10)
        returned_ids = [r["nct_id"] for r in results]
        assert WALKTHROUGH_NCT_ID in returned_ids, (
            f"Expected {WALKTHROUGH_NCT_ID} in top 10, got: {returned_ids}"
        )

    def test_b3_disease_type_separation(self, collection):
        """
        A prostate cancer query should not surface gynecologic trials in the top 5.
        Cancer type should be the dominant retrieval signal.
        """
        query = (
            "Male patient with metastatic castration-resistant prostate cancer, "
            "rising PSA after androgen deprivation therapy"
        )
        vec = _embed_one(query)
        results = query_trials(collection, vec, n_results=5)
        gynecologic_keywords = {"ovarian", "cervical", "uterine", "endometrial",
                                 "fallopian", "vulvar", "gynecol"}
        for r in results:
            cond_lower = r["conditions"].lower()
            matched = [kw for kw in gynecologic_keywords if kw in cond_lower]
            assert not matched, (
                f"Prostate query returned gynecologic trial {r['nct_id']} "
                f"(conditions: {r['conditions'][:80]})"
            )

    def test_b4_recurrent_vs_naive_differentiation(self, collection):
        """
        Retrieval sets for treatment-naive vs heavily pretreated patients
        should be substantially different (Jaccard < 0.5).
        """
        naive_query = (
            "Newly diagnosed ovarian cancer, treatment naive, no prior chemotherapy"
        )
        recurrent_query = (
            "Platinum-resistant recurrent ovarian cancer, 3 or more prior lines of therapy"
        )
        naive_vec = _embed_one(naive_query)
        recurrent_vec = _embed_one(recurrent_query)

        naive_ids = {r["nct_id"] for r in query_trials(collection, naive_vec, n_results=10)}
        recurrent_ids = {r["nct_id"] for r in query_trials(collection, recurrent_vec, n_results=10)}

        jaccard = _jaccard(naive_ids, recurrent_ids)
        assert jaccard < 0.5, (
            f"Expected Jaccard < 0.5 between naive and recurrent queries, got {jaccard:.2f}. "
            f"Overlap: {naive_ids & recurrent_ids}"
        )

    def test_b5_metadata_filter_status(self, collection):
        """
        Applying a status filter should return only trials matching that status.
        Tests the ChromaDB where-clause path in query_trials().
        """
        query = (
            "Newly diagnosed female patient with stage III ovarian carcinoma, "
            "no prior chemotherapy"
        )
        vec = _embed_one(query)
        results = query_trials(
            collection, vec, n_results=10,
            filters={"status": {"$eq": "RECRUITING"}}
        )
        assert len(results) > 0, "Filter returned no results — check RECRUITING trials exist"
        for r in results:
            assert r["status"] == "RECRUITING", (
                f"Filter violation: {r['nct_id']} has status={r['status']}"
            )

    def test_b6_score_floor(self, collection):
        """Top result for any reasonable oncology query should have score > 0.5."""
        query = "Adult patient with solid tumor malignancy"
        vec = _embed_one(query)
        results = query_trials(collection, vec, n_results=1)
        assert len(results) > 0, "Query returned no results"
        top_score = results[0]["score"]
        assert top_score > 0.5, (
            f"Top result score {top_score:.3f} below floor of 0.5 — "
            "collection may be malformed or embeddings zeroed"
        )

    def test_b7_biomarker_specificity(self, collection):
        """
        HER2+ breast cancer query should not return ovarian or prostate trials
        in the top 5. Biomarker + cancer type together should constrain retrieval.
        """
        query = (
            "HER2-positive metastatic breast cancer, trastuzumab naive, "
            "hormone receptor negative"
        )
        vec = _embed_one(query)
        results = query_trials(collection, vec, n_results=5)
        excluded_keywords = {"ovarian", "prostate", "cervical", "lung", "colorectal"}
        for r in results:
            cond_lower = r["conditions"].lower()
            matched = [kw for kw in excluded_keywords if kw in cond_lower]
            assert not matched, (
                f"HER2+ breast query returned off-target trial {r['nct_id']} "
                f"(conditions: {r['conditions'][:80]})"
            )
