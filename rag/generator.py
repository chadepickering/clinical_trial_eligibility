"""
Response generation via Mistral-7B served through Ollama.

Design: per-trial eligibility Q&A.

The LLM is asked one focused question per trial:
    "Based on these eligibility criteria, is this patient eligible?
    Answer ELIGIBLE, NOT ELIGIBLE, or UNCERTAIN with a brief explanation."

This keeps the context window requirement small (~600-800 tokens per call),
produces interpretable per-criterion output, and integrates cleanly with the
Bayesian scorer which also operates on a per-trial basis.

The Ollama HTTP API is used directly via requests — no additional LLM SDK
dependency. The endpoint is stable and the integration is transparent.

Usage:
    from rag.generator import build_prompt, generate, assess_trial

    result = assess_trial(
        nct_id="NCT02222883",
        trial_document="...",   # composite text from ChromaDB
        patient_query="52yo female, BRCA1 mutation, platinum-sensitive recurrent OC",
    )
    # result: {"nct_id": ..., "verdict": "ELIGIBLE|NOT ELIGIBLE|UNCERTAIN",
    #          "explanation": ..., "raw": ...}
"""

import json
import re

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "mistral"
DEFAULT_TIMEOUT = 120  # seconds

SYSTEM_PROMPT = (
    "You are a clinical research coordinator reviewing oncology clinical trial "
    "eligibility criteria. You are given a patient description and the eligibility "
    "criteria for one trial. Your task is to assess whether the patient is likely "
    "eligible for the trial based solely on the information provided.\n\n"
    "Rules:\n"
    "- Base your assessment only on the eligibility criteria provided.\n"
    "- If a criterion cannot be determined from the patient description, "
    "treat it as uncertain rather than assuming met or not met.\n"
    "- Be concise. Explain your verdict in 2-4 sentences, citing specific criteria.\n"
    "- End your response with exactly one of these verdicts on its own line:\n"
    "  VERDICT: ELIGIBLE\n"
    "  VERDICT: NOT ELIGIBLE\n"
    "  VERDICT: UNCERTAIN"
)

# ---------------------------------------------------------------------------
# Few-shot examples
#
# Three worked examples demonstrating the numeric/temporal comparison failures
# observed in Step 9 evaluation (7/50 ineligible cases received ELIGIBLE).
# All use fictional NCT IDs (NCT00000001–3) to avoid ambiguity with real trials.
#
# Pattern covered by each example:
#   1. Numeric lab threshold  — platelet count below required minimum
#   2. Age range (lower bound) — patient below minimum age
#   3. Time window (temporal)  — platinum-sensitive misclassified as resistant
#
# Token overhead: ~265 tokens. Revised document budget: 3,080 tokens ≈ 12,320
# chars. Default doc_max_chars=12,000 remains within budget.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """\
---
EXAMPLES (for reference — do not assess these, use them as a guide):

TRIAL: NCT00000001
Inclusion Criteria: Platelet count ≥ 100,000/mm³.

PATIENT: Female, 44yo. Primary breast cancer. Platelet count 78,000/mm³. ECOG 0.

Assessment: The trial requires platelet count ≥ 100,000/mm³. The patient's platelet count is 78,000/mm³. Since 78,000 < 100,000, this inclusion criterion is not met.
VERDICT: NOT ELIGIBLE

---
TRIAL: NCT00000002
Age eligibility: 65 Years and older.
Inclusion Criteria: Age ≥ 65 years.

PATIENT: Female, 59yo. Stage II breast cancer survivor, post-treatment.

Assessment: The trial requires age ≥ 65 years. The patient is 59 years old. Since 59 < 65, the minimum age criterion is not met.
VERDICT: NOT ELIGIBLE

---
TRIAL: NCT00000003
Inclusion Criteria: Platinum-resistant disease, defined as disease progression within 6 months of completing platinum-based chemotherapy.

PATIENT: Female, 55yo. Recurrent ovarian carcinoma. Progressed 14 months after last platinum-based chemotherapy.

Assessment: The trial requires platinum-resistant disease (progression < 6 months after platinum therapy). The patient progressed at 14 months. Since 14 > 6, the disease is platinum-sensitive, not platinum-resistant. This inclusion criterion is not met.
VERDICT: NOT ELIGIBLE

"""


def build_prompt(
    nct_id: str,
    trial_document: str,
    patient_query: str,
    doc_max_chars: int = 12_000,
) -> str:
    """
    Construct the full prompt string for a single trial eligibility assessment.

    Token budget (Mistral 4096-token context window):
        Fixed overhead (system prompt + few-shot examples + patient query
                        + scaffolding):                                 ~504 tokens
        Reserve for generated output:                                   512 tokens
        Available for trial document:                                 3,080 tokens
                                                                   ≈ 12,320 chars

    Default of 12,000 chars fits within the revised budget (12,000 < 12,320).
    The previous default of 1,500 chars cut off the eligibility criteria section
    entirely for most trials. Few-shot examples (~265 tokens) were added in Step 9
    to improve numeric threshold reasoning; they reduce the document budget by ~265
    tokens vs the original 3,345 but remain within the 12,000-char default.

    Args:
        nct_id:         trial identifier, included in the prompt for traceability
        trial_document: composite trial text from ChromaDB (title + summary + eligibility)
        patient_query:  free-text patient description
        doc_max_chars:  character cap on the trial document in the prompt

    Returns:
        formatted prompt string
    """
    truncated_doc = trial_document[:doc_max_chars]
    if len(trial_document) > doc_max_chars:
        truncated_doc += "\n[... eligibility criteria truncated ...]"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        f"---\n"
        f"TRIAL: {nct_id}\n\n"
        f"{truncated_doc}\n\n"
        f"---\n"
        f"PATIENT: {patient_query}\n\n"
        f"Assessment:"
    )


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float | None = None,
) -> str:
    """
    Send a prompt to the Ollama HTTP API and return the generated text.

    Uses stream=False so the full response arrives in a single JSON object.

    Args:
        prompt:      full prompt string (system + context + query)
        model:       Ollama model name (must be pulled: ollama pull <model>)
        timeout:     request timeout in seconds
        temperature: sampling temperature passed to Ollama options.
                     None (default) uses Ollama's server default (~0.7).
                     0.0 enables greedy decoding for deterministic outputs —
                     used by rag/evaluate.py to make evaluation reproducible.

    Returns:
        generated text string

    Raises:
        requests.exceptions.ConnectionError: if Ollama server is not running
        requests.exceptions.Timeout:         if generation exceeds timeout
        ValueError:                          if Ollama returns a non-200 status
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)

    if response.status_code != 200:
        raise ValueError(
            f"Ollama returned status {response.status_code}: {response.text[:200]}"
        )

    return response.json()["response"]


def _parse_verdict(raw: str) -> str:
    """
    Extract the VERDICT line from the generated text.

    Returns "ELIGIBLE", "NOT ELIGIBLE", or "UNCERTAIN".
    Falls back to "UNCERTAIN" if no verdict line is found.
    """
    match = re.search(
        r"VERDICT:\s*(ELIGIBLE|NOT ELIGIBLE|UNCERTAIN)",
        raw,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return "UNCERTAIN"


def assess_trial(
    nct_id: str,
    trial_document: str,
    patient_query: str,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float | None = None,
) -> dict:
    """
    Assess whether a patient is eligible for a single trial.

    Builds the prompt, calls Ollama, parses the verdict, and returns a
    structured result dict. This is the primary entry point for the pipeline.

    Args:
        nct_id:         trial identifier
        trial_document: composite trial text from ChromaDB
        patient_query:  free-text patient description
        model:          Ollama model name
        timeout:        generation timeout in seconds
        temperature:    sampling temperature (None = Ollama default ~0.7;
                        0.0 = greedy/deterministic, used by evaluate.py)

    Returns:
        dict with keys:
            nct_id      — trial identifier
            verdict     — "ELIGIBLE", "NOT ELIGIBLE", or "UNCERTAIN"
            explanation — the full generated text (includes verdict line)
            raw         — alias for explanation (convenience for downstream use)
    """
    prompt = build_prompt(nct_id, trial_document, patient_query)
    raw = generate(prompt, model=model, timeout=timeout, temperature=temperature)
    verdict = _parse_verdict(raw)

    return {
        "nct_id":      nct_id,
        "verdict":     verdict,
        "explanation": raw.strip(),
        "raw":         raw.strip(),
    }
