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
    "- Base your assessment only on the eligibility criteria explicitly stated "
    "in the trial document. Do not add criteria from your medical knowledge or "
    "assume a criterion exists if it is not written in the trial document.\n"
    "- Use only the patient facts explicitly stated in the patient description. "
    "Do not infer, assume, or fabricate patient history, lab values, diagnoses, "
    "or treatment history that are not directly stated.\n"
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
# All use fictional NCT IDs (NCT00000001–9) to avoid ambiguity with real trials.
#
# Pattern covered by each example:
#   1.  Numeric lab threshold FAIL  — platelet count below required minimum
#   2.  Age range FAIL              — patient below minimum age
#   3.  Time window FAIL            — platinum-sensitive misclassified as resistant
#   4.  Numeric lab threshold PASS  — comma-formatted ANC correctly above threshold
#   5.  Performance status FAIL     — ECOG above maximum allowed
#   6.  Prior therapy exclusion     — anthracycline exposure disqualifies patient
#   7.  Organ function FAIL         — bilirubin exceeds hepatic function limit
#   8.  Missing information         — key criterion unverifiable from description → UNCERTAIN
#   9.  Fully eligible              — all inclusion criteria clearly met across multiple domains
#   10. Cancer type mismatch FAIL   — patient's cancer type differs from trial's required diagnosis
#   11. Prior treatment required    — trial requires ≥1 prior line; treatment-naïve patient ineligible
#
# Note: Mistral-7B-Instruct-v0.2 supports a 32K context window, so the expanded
# few-shot block (~750 tokens) leaves ample room for trial documents at the
# default doc_max_chars=12,000 (~3,000 tokens).
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

---
TRIAL: NCT00000004
Inclusion Criteria: Total granulocyte count > 2,000/mm³. Platelet count > 100,000/mm³.

PATIENT: Female, 52yo. Ovarian carcinoma. ANC 2,800/mm³. Platelet count 180,000/mm³.

Assessment: The trial requires total granulocyte count > 2,000/mm³. The patient's ANC is 2,800/mm³. Since 2,800 > 2,000, this criterion is met. The trial requires platelet count > 100,000/mm³. The patient's platelet count is 180,000/mm³. Since 180,000 > 100,000, this criterion is also met.
VERDICT: ELIGIBLE

---
TRIAL: NCT00000005
Inclusion Criteria: ECOG performance status ≤ 2.

PATIENT: Male, 67yo. Metastatic non-small cell lung cancer. ECOG performance status 3.

Assessment: The trial requires ECOG performance status ≤ 2. The patient's ECOG is 3. Since 3 > 2, the performance status criterion is not met.
VERDICT: NOT ELIGIBLE

---
TRIAL: NCT00000006
Exclusion Criteria: Prior treatment with any anthracycline-based chemotherapy (e.g., doxorubicin, epirubicin).

PATIENT: Female, 48yo. Recurrent breast cancer. Prior regimens: AC (doxorubicin + cyclophosphamide) followed by paclitaxel.

Assessment: The trial excludes patients with prior anthracycline-based chemotherapy. The patient received doxorubicin (AC regimen), which is an anthracycline. This exclusion criterion is triggered; the patient is ineligible.
VERDICT: NOT ELIGIBLE

---
TRIAL: NCT00000007
Inclusion Criteria: Adequate hepatic function: total bilirubin ≤ 1.5 × upper limit of normal (ULN). AST and ALT ≤ 3 × ULN.

PATIENT: Female, 61yo. Hepatocellular carcinoma. Total bilirubin 3.2 mg/dL (ULN 1.0 mg/dL). AST 52 U/L (ULN 40 U/L). ALT 38 U/L.

Assessment: The trial requires total bilirubin ≤ 1.5 × ULN (≤ 1.5 mg/dL). The patient's bilirubin is 3.2 mg/dL. Since 3.2 > 1.5, this hepatic function criterion is not met. AST 52 vs limit 3 × 40 = 120: met. However, the bilirubin failure alone disqualifies the patient.
VERDICT: NOT ELIGIBLE

---
TRIAL: NCT00000008
Inclusion Criteria: Adequate pulmonary function: FEV1 ≥ 60% of predicted and DLCO ≥ 50% of predicted.

PATIENT: Female, 62yo. Non-small cell lung cancer, Stage IIIA. ECOG 1. Prior lobectomy.

Assessment: The trial requires FEV1 ≥ 60% predicted and DLCO ≥ 50% predicted. The patient description does not include pulmonary function test results. Since these criteria cannot be verified from the information provided, eligibility is uncertain.
VERDICT: UNCERTAIN

---
TRIAL: NCT00000009
Inclusion Criteria: Histologically confirmed ovarian carcinoma, FIGO Stage III or IV. No prior chemotherapy or radiation. ECOG ≤ 2. Hemoglobin ≥ 9.0 g/dL. Creatinine ≤ 1.5 mg/dL.

PATIENT: Female, 56yo. Stage IV ovarian serous carcinoma, biopsy confirmed. No prior chemotherapy or radiation therapy. ECOG 1. Hemoglobin 11.2 g/dL. Creatinine 0.9 mg/dL.

Assessment: Histologically confirmed ovarian carcinoma: biopsy confirmed. Stage III or IV: Stage IV, met. No prior chemotherapy or radiation: met. ECOG ≤ 2: patient ECOG 1, met. Hemoglobin ≥ 9.0 g/dL: 11.2 g/dL, met. Creatinine ≤ 1.5 mg/dL: 0.9 mg/dL, met. All stated criteria are satisfied.
VERDICT: ELIGIBLE

---
TRIAL: NCT00000010
Inclusion Criteria: Histologically documented recurrent or refractory endometrial adenocarcinoma or uterine sarcoma. No prior chemotherapy for metastatic disease. ECOG ≤ 2.

PATIENT: Female, 52yo. Stage III ovarian serous carcinoma. No prior chemotherapy or radiotherapy. ECOG 1.

Assessment: The trial requires histologically documented endometrial adenocarcinoma or uterine sarcoma. The patient has ovarian serous carcinoma, which is a distinct histologic entity and does not meet this disease-specific inclusion criterion. The cancer type stated in the trial document does not match the patient's diagnosis. This inclusion criterion is not met regardless of other characteristics.
VERDICT: NOT ELIGIBLE

---
TRIAL: NCT00000011
Inclusion Criteria: At least one prior line of platinum-based chemotherapy for advanced disease. Measurable disease per RECIST 1.1 criteria. ECOG ≤ 2.

PATIENT: Female, 44yo. Stage IIB ovarian clear cell carcinoma. No prior chemotherapy or radiotherapy. ECOG 0.

Assessment: The trial requires at least one prior line of platinum-based chemotherapy. The patient has received no prior chemotherapy. Since the patient has not received any prior chemotherapy, this inclusion criterion is not met. The absence of prior treatment disqualifies the patient regardless of all other characteristics being met.
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
