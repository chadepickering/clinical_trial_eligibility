"""
Generator tests.

Three test classes:

  TestGeneratorContract    — prompt construction and verdict parsing; no Ollama
                             dependency. Always runnable.
  TestGeneratorLive        — end-to-end calls to a running Ollama server. Skipped
                             automatically if Ollama is unreachable or mistral is
                             not pulled.
  TestIneligibilityVerdict — 8 clearly ineligible patient profiles each with one
                             hard, objective disqualifying criterion. Requires Ollama
                             and ChromaDB. Asserts that no clearly ineligible patient
                             receives an ELIGIBLE verdict (UNCERTAIN is acceptable).

                             These tests document the known limitation of Mistral-7B:
                             it frequently returns UNCERTAIN rather than NOT ELIGIBLE
                             when it identifies a disqualifying criterion but also
                             observes uncertainty elsewhere. UNCERTAIN is acceptable;
                             ELIGIBLE is not.

Run all:
    PYTHONPATH=. pytest tests/test_generator.py -v

Skip live tests:
    PYTHONPATH=. pytest tests/test_generator.py -v -k "not Live and not Ineligibility"
"""

import time
from unittest.mock import patch

import pytest
import requests

from rag.generator import (
    OLLAMA_URL,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    build_prompt,
    generate,
    assess_trial,
    _parse_verdict,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALKTHROUGH_NCT_ID = "NCT00127920"

PATIENT_ELIGIBLE = (
    "Female patient, 54 years old. Histologically confirmed stage III "
    "ovarian carcinoma. No prior chemotherapy or radiotherapy. Karnofsky "
    "performance status 80%. Adequate bone marrow and hepatic function."
)

PATIENT_INELIGIBLE = (
    "Female patient, 61 years old. Recurrent stage III ovarian carcinoma. "
    "Three prior lines of chemotherapy including carboplatin and paclitaxel. "
    "Karnofsky performance status 70%."
)

SHORT_TRIAL_DOC = (
    "Pilot Study of Taxol, Carboplatin, and Bevacizumab in Advanced Stage "
    "Ovarian Carcinoma Patients. Inclusion Criteria: stage III/IV ovarian "
    "cancer, no prior chemotherapy, Karnofsky >50%. Exclusion Criteria: "
    "prior bevacizumab, severe cardiac history."
)

VALID_VERDICTS = {"ELIGIBLE", "NOT ELIGIBLE", "UNCERTAIN"}

# Eight clearly ineligible patients, each with exactly one hard disqualifier.
# (case_id, disqualifier_note, patient_description)
INELIGIBLE_CASES = [
    (
        "prior_chemo",
        "Three prior lines of chemo — violates 'no prior chemotherapy' inclusion criterion",
        "Female, 61yo. Recurrent stage III ovarian carcinoma. Three prior lines of "
        "chemotherapy including carboplatin and paclitaxel. Karnofsky 70%.",
    ),
    (
        "prior_radiotherapy",
        "Prior pelvic radiotherapy — violates 'no prior radiotherapy' inclusion criterion",
        "Female, 58yo. Stage III ovarian carcinoma. Prior pelvic radiotherapy completed "
        "8 months ago. No prior chemotherapy. Karnofsky 80%.",
    ),
    (
        "low_karnofsky",
        "Karnofsky 40% — below the required >50% inclusion criterion",
        "Female, 66yo. Newly diagnosed stage IV ovarian carcinoma. No prior chemotherapy "
        "or radiotherapy. Karnofsky performance status 40%. Adequate hepatic function.",
    ),
    (
        "low_malignancy_potential",
        "Epithelial OC of low malignancy potential — explicit exclusion criterion",
        "Female, 49yo. Epithelial ovarian cancer of low malignancy potential (borderline "
        "tumor). No prior chemotherapy. Karnofsky 90%.",
    ),
    (
        "septicemia",
        "Active septicemia — explicit exclusion criterion",
        "Female, 55yo. Stage III ovarian carcinoma. No prior chemotherapy. Currently "
        "hospitalised with septicemia and systemic infection. Karnofsky 60%.",
    ),
    (
        "cardiac_history",
        "MI within past 6 months — explicit exclusion criterion",
        "Female, 63yo. Newly diagnosed stage IV ovarian carcinoma. No prior chemotherapy. "
        "Myocardial infarction 3 months ago. Karnofsky 75%.",
    ),
    (
        "wrong_cancer_type",
        "Endometrial cancer — trial restricted to ovarian/fallopian/peritoneal cancer",
        "Female, 57yo. Stage III endometrial carcinoma. No prior chemotherapy or "
        "radiotherapy. Karnofsky 85%. Adequate organ function.",
    ),
    (
        "male_patient",
        "Male patient — trial sex eligibility is FEMALE",
        "Male, 54yo. Stage III peritoneal carcinoma. No prior chemotherapy or "
        "radiotherapy. Karnofsky 80%. Adequate organ function.",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ollama_available():
    """Skip the live test class if Ollama is not reachable or mistral not pulled."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any("mistral" in m for m in models):
            pytest.skip("mistral model not pulled — run: ollama pull mistral")
    except requests.exceptions.ConnectionError:
        pytest.skip("Ollama server not reachable — run: ollama serve")
    except requests.exceptions.Timeout:
        pytest.skip("Ollama server timed out")


@pytest.fixture(scope="module")
def walkthrough_doc(ollama_available):
    """Fetch NCT00127920's composite document from ChromaDB."""
    from rag.vector_store import get_client, get_collection, collection_count
    client = get_client("data/processed/chroma")
    col = get_collection(client)
    if collection_count(col) == 0:
        pytest.skip("ChromaDB collection is empty — run embed.py first")
    result = col.get(ids=[WALKTHROUGH_NCT_ID], include=["documents"])
    return result["documents"][0]


# ---------------------------------------------------------------------------
# E — Generator contract (no Ollama)
# ---------------------------------------------------------------------------

class TestGeneratorContract:

    # --- build_prompt ---

    def test_e1_prompt_contains_nct_id(self):
        """NCT ID must appear in the prompt for traceability."""
        prompt = build_prompt("NCT99999999", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE)
        assert "NCT99999999" in prompt

    def test_e2_prompt_contains_patient_query(self):
        """Patient description must appear verbatim in the prompt."""
        prompt = build_prompt("NCT99999999", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE)
        assert PATIENT_ELIGIBLE in prompt

    def test_e3_prompt_contains_system_prompt(self):
        """System prompt must be present — it carries the assessment rules."""
        prompt = build_prompt("NCT99999999", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE)
        # Check a distinctive phrase from the system prompt rather than the
        # full string, in case whitespace normalisation differs.
        assert "VERDICT:" in prompt
        assert "clinical research coordinator" in prompt

    def test_e4_prompt_contains_trial_doc_when_short(self):
        """Short documents must appear in full, not truncated."""
        prompt = build_prompt("NCT99999999", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE,
                               doc_max_chars=5000)
        assert SHORT_TRIAL_DOC in prompt

    def test_e5_prompt_truncates_long_document(self):
        """
        Documents longer than doc_max_chars must be truncated and the
        truncation marker appended. Without this, long eligibility texts
        overflow the Ollama context window.
        """
        long_doc = "A" * 3000
        prompt = build_prompt("NCT99999999", long_doc, PATIENT_ELIGIBLE,
                               doc_max_chars=100)
        assert "A" * 100 in prompt
        assert "A" * 101 not in prompt
        assert "truncated" in prompt.lower()

    def test_e6_prompt_default_truncation_is_12000_chars(self):
        """
        Default doc_max_chars is 12,000 — covers p99 of the corpus while
        staying within Mistral's 4096-token context window.

        Verify by supplying a document longer than 12,000 chars and checking
        that the default prompt is shorter than a full-document prompt and
        contains the truncation marker.
        """
        long_doc = "X" * 15000
        prompt_default = build_prompt("NCT99999999", long_doc, PATIENT_ELIGIBLE)
        prompt_full    = build_prompt("NCT99999999", long_doc, PATIENT_ELIGIBLE,
                                      doc_max_chars=15000)
        # Default prompt must be shorter than full-doc prompt.
        assert len(prompt_default) < len(prompt_full)
        # And must contain the truncation marker.
        assert "truncated" in prompt_default.lower()

    # --- _parse_verdict ---

    def test_e7_parse_verdict_eligible(self):
        assert _parse_verdict("Some reasoning.\nVERDICT: ELIGIBLE") == "ELIGIBLE"

    def test_e8_parse_verdict_not_eligible(self):
        assert _parse_verdict("Reasoning.\nVERDICT: NOT ELIGIBLE") == "NOT ELIGIBLE"

    def test_e9_parse_verdict_uncertain(self):
        assert _parse_verdict("Reasoning.\nVERDICT: UNCERTAIN") == "UNCERTAIN"

    def test_e10_parse_verdict_case_insensitive(self):
        """The regex must match regardless of case in the generated output."""
        assert _parse_verdict("verdict: eligible") == "ELIGIBLE"
        assert _parse_verdict("VERDICT: not eligible") == "NOT ELIGIBLE"

    def test_e11_parse_verdict_fallback_to_uncertain(self):
        """
        If the model omits the VERDICT line entirely (e.g. hallucination or
        truncation), _parse_verdict must fall back to UNCERTAIN rather than
        returning None or raising.
        """
        result = _parse_verdict("The patient may or may not qualify.")
        assert result == "UNCERTAIN"

    def test_e12_parse_verdict_ignores_verdict_in_body_text(self):
        """
        The word VERDICT may appear in the explanation body. Only the
        structured 'VERDICT: <label>' line should be matched.
        This test confirms a verdict label embedded in prose is not
        incorrectly extracted as the final verdict.
        """
        raw = (
            "The verdict is unclear because several criteria cannot be assessed. "
            "A definitive verdict requires more lab values.\n"
            "VERDICT: UNCERTAIN"
        )
        assert _parse_verdict(raw) == "UNCERTAIN"

    # --- assess_trial (mocked generate) ---

    def test_e13_assess_trial_returns_required_keys(self):
        """
        assess_trial must return a dict with nct_id, verdict, explanation,
        and raw — regardless of what the model generates.
        """
        with patch("rag.generator.generate", return_value="Some text.\nVERDICT: ELIGIBLE"):
            result = assess_trial("NCT99999999", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE)

        assert set(result.keys()) >= {"nct_id", "verdict", "explanation", "raw"}

    def test_e14_assess_trial_preserves_nct_id(self):
        """nct_id in the result must match what was passed in."""
        with patch("rag.generator.generate", return_value="Text.\nVERDICT: UNCERTAIN"):
            result = assess_trial("NCT12345678", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE)

        assert result["nct_id"] == "NCT12345678"

    def test_e15_assess_trial_verdict_matches_parsed_output(self):
        """The verdict field must reflect what _parse_verdict extracts from raw."""
        with patch("rag.generator.generate", return_value="Analysis.\nVERDICT: NOT ELIGIBLE"):
            result = assess_trial("NCT99999999", SHORT_TRIAL_DOC, PATIENT_INELIGIBLE)

        assert result["verdict"] == "NOT ELIGIBLE"

    def test_e16_assess_trial_explanation_is_non_empty_string(self):
        """explanation and raw must be non-empty strings after stripping."""
        with patch("rag.generator.generate", return_value="  Analysis.\nVERDICT: ELIGIBLE  "):
            result = assess_trial("NCT99999999", SHORT_TRIAL_DOC, PATIENT_ELIGIBLE)

        assert isinstance(result["explanation"], str) and result["explanation"]
        assert isinstance(result["raw"], str) and result["raw"]

    def test_e17_generate_raises_on_bad_status(self):
        """
        generate() must raise ValueError when Ollama returns a non-200 status,
        not silently return empty string or None.
        """
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "Internal Server Error"
            with pytest.raises(ValueError, match="500"):
                generate("test prompt")


# ---------------------------------------------------------------------------
# F — Generator live (requires Ollama + mistral)
# ---------------------------------------------------------------------------

class TestGeneratorLive:

    def test_f1_eligible_patient_returns_eligible(self, ollama_available, walkthrough_doc):
        """
        8c smoke test #1. A patient who meets all criteria for NCT00127920
        (stage III OC, no prior chemo, adequate function) should receive
        an ELIGIBLE verdict.
        """
        result = assess_trial(WALKTHROUGH_NCT_ID, walkthrough_doc, PATIENT_ELIGIBLE)
        assert result["verdict"] == "ELIGIBLE", (
            f"Expected ELIGIBLE for clearly eligible patient, got {result['verdict']}.\n"
            f"Explanation: {result['explanation']}"
        )

    def test_f2_ineligible_patient_not_eligible_or_uncertain(self, ollama_available, walkthrough_doc):
        """
        8c smoke test #2. A patient with three prior lines of chemotherapy
        is excluded by NCT00127920's 'no prior chemotherapy' criterion.
        Mistral-7B observed to return UNCERTAIN rather than NOT ELIGIBLE
        for this case — both are acceptable; ELIGIBLE is not.

        If this test fails (verdict = ELIGIBLE), the LLM is failing to
        apply the most basic eligibility exclusion.
        """
        result = assess_trial(WALKTHROUGH_NCT_ID, walkthrough_doc, PATIENT_INELIGIBLE)
        assert result["verdict"] != "ELIGIBLE", (
            f"Expected NOT ELIGIBLE or UNCERTAIN for patient with prior chemo, "
            f"got ELIGIBLE.\nExplanation: {result['explanation']}"
        )

    def test_f3_verdict_is_always_a_valid_value(self, ollama_available, walkthrough_doc):
        """
        The verdict field must always be one of the three defined values,
        never None, empty, or a free-form string. Tests both patients to
        exercise both verdict paths.
        """
        for patient in (PATIENT_ELIGIBLE, PATIENT_INELIGIBLE):
            result = assess_trial(WALKTHROUGH_NCT_ID, walkthrough_doc, patient)
            assert result["verdict"] in VALID_VERDICTS, (
                f"Invalid verdict '{result['verdict']}' for patient: {patient[:60]}"
            )

    def test_f4_warm_latency_under_60s(self, ollama_available, walkthrough_doc):
        """
        Warm-path latency (model already loaded) must be under 60 seconds.
        The model loads on f1/f2; by f4 it is resident in VRAM.

        60s is intentionally generous — the acceptance criterion of <30s
        is for the full pipeline (8d), not generation alone. This test
        catches pathological slowdowns (e.g. model evicted from VRAM,
        CPU fallback) without being brittle to normal variance.
        """
        t0 = time.time()
        assess_trial(WALKTHROUGH_NCT_ID, walkthrough_doc, PATIENT_ELIGIBLE)
        elapsed = time.time() - t0
        assert elapsed < 60, (
            f"Generation took {elapsed:.1f}s — model may have been evicted from VRAM"
        )

    def test_f5_explanation_mentions_trial_criteria(self, ollama_available, walkthrough_doc):
        """
        The explanation for the eligible patient must reference at least one
        specific clinical concept from the trial criteria (performance status,
        chemotherapy, ovarian, carcinoma, or bevacizumab). A purely generic
        response indicates the model ignored the provided context.
        """
        result = assess_trial(WALKTHROUGH_NCT_ID, walkthrough_doc, PATIENT_ELIGIBLE)
        clinical_terms = {
            "chemotherapy", "karnofsky", "ovarian", "carcinoma",
            "bevacizumab", "performance", "bone marrow", "hepatic",
        }
        explanation_lower = result["explanation"].lower()
        matched = [t for t in clinical_terms if t in explanation_lower]
        assert matched, (
            f"Explanation contains no clinical trial terms — model may have "
            f"ignored context.\nExplanation: {result['explanation']}"
        )


# ---------------------------------------------------------------------------
# G — Ineligibility verdict distribution (requires Ollama + ChromaDB)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def walkthrough_doc_chroma():
    """
    Fetch NCT00127920's composite document from ChromaDB.
    Separate from the walkthrough_doc fixture used in TestGeneratorLive
    to avoid coupling fixture dependencies across classes.
    """
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any("mistral" in m for m in models):
            pytest.skip("mistral model not pulled — run: ollama pull mistral")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip("Ollama server not reachable — run: ollama serve")

    from rag.vector_store import get_client, get_collection, collection_count
    client = get_client("data/processed/chroma")
    col = get_collection(client)
    if collection_count(col) == 0:
        pytest.skip("ChromaDB collection is empty — run embed.py first")
    result = col.get(ids=[WALKTHROUGH_NCT_ID], include=["documents"])
    return result["documents"][0]


class TestIneligibilityVerdict:
    """
    Evaluates verdict distribution across 8 clearly ineligible patients,
    each with one hard objective disqualifying criterion for NCT00127920.

    Assertion: no clearly ineligible patient receives ELIGIBLE.
    UNCERTAIN is the documented expected output for most cases — Mistral-7B
    hedges when it identifies a disqualifying criterion but observes
    uncertainty elsewhere. This is preferable to ELIGIBLE. The Bayesian
    scorer (Step 10) handles the rigorous eligibility determination.

    Observed distribution after eligibility header + doc_max_chars=12000:
        NOT ELIGIBLE: 1/8  (prior_chemo)
        UNCERTAIN:    7/8
        ELIGIBLE:     0/8
    """

    @pytest.mark.parametrize("case_id,disqualifier,patient", INELIGIBLE_CASES)
    def test_g_ineligible_patient_not_eligible(
        self, case_id, disqualifier, patient, walkthrough_doc_chroma
    ):
        """
        Assert that a clearly ineligible patient does not receive ELIGIBLE.

        Each case has one hard disqualifying criterion that should be visible
        in the trial document (inclusion/exclusion text or eligibility header).
        The hard assertion is: verdict != ELIGIBLE.
        NOT ELIGIBLE and UNCERTAIN are both acceptable.

        The disqualifier note in the parametrize list documents which specific
        criterion is violated, for diagnostic clarity on failure.
        """
        result = assess_trial(WALKTHROUGH_NCT_ID, walkthrough_doc_chroma, patient)
        assert result["verdict"] != "ELIGIBLE", (
            f"[{case_id}] Patient with hard disqualifier '{disqualifier}' "
            f"received ELIGIBLE verdict.\n"
            f"Explanation: {result['explanation']}"
        )
