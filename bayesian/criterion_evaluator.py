"""
Patient-vs-criterion matching for the Bayesian eligibility model.

Design: keyword-routed numeric comparison for objective+observable criteria;
heuristic hedging estimation for subjective criteria; DuckDB loader that
preserves NULL labels so the Bayesian model can marginalize over them.

Patient profile schema (flat dict):
    age           (int)
    sex           (str: "male" | "female")
    ecog          (int: 0–4)
    karnofsky     (int: 0–100)
    cancer_type   (str, free-text)
    prior_chemo   (bool)
    prior_rt      (bool)
    lab_values    (dict[str, float]):
        platelet_count    — /mm³
        hemoglobin        — g/dL
        neutrophil_count  — /mm³
        creatinine        — mg/dL
        bilirubin         — mg/dL
        alt               — U/L
        ast               — U/L
        lvef              — %
        testosterone      — ng/dL

For criteria where the patient profile lacks the relevant field,
evaluate_objective_criterion returns None — the Bayesian model treats
these as uncertain (Beta(1,1) prior) regardless of B2/B3 label.

Usage:
    from bayesian.criterion_evaluator import (
        Criterion, load_criteria_for_trial,
        evaluate_objective_criterion, estimate_hedging,
    )

    import duckdb
    con = duckdb.connect("data/processed/trials.duckdb")
    criteria = load_criteria_for_trial("NCT00127920", con)
    patient = {"age": 52, "sex": "female", "ecog": 1, "karnofsky": 80,
               "lab_values": {"platelet_count": 150_000}}
    for c in criteria:
        result = evaluate_objective_criterion(c, patient)
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Criterion dataclass
# ---------------------------------------------------------------------------

@dataclass
class Criterion:
    criterion_id: str
    text: str
    b1_label: int | None           # 1=inclusion, 0=exclusion, None=unknown
    b2_label: int | None           # 1=objective, 0=subjective, None=unknown
    b3_label: int | None           # 1=observable, 0=unobservable, None=unknown
    b2_confidence: float = 0.0
    b3_confidence: float = 0.0
    extracted_thresholds: list[str] = field(default_factory=list)
    extracted_demographics: list[str] = field(default_factory=list)
    extracted_lab_values: list[str] = field(default_factory=list)
    extracted_conditions: list[str] = field(default_factory=list)
    extracted_drugs: list[str] = field(default_factory=list)
    extracted_scales: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Threshold parsing
# ---------------------------------------------------------------------------

_THRESHOLD_RE = re.compile(
    r"([≥≤><]=?|>=|<=)\s*(\d[\d,]*\.?\d*)\s*([a-zA-Z%/µ³²°×·\s\d]*)",
    re.UNICODE,
)

_OP_MAP = {
    "≥": ">=", ">=": ">=",
    "≤": "<=", "<=": "<=",
    ">": ">",
    "<": "<",
}


def _parse_threshold(s: str) -> tuple[str, float, str] | None:
    """
    Parse a threshold string into (operator, value, unit).

    Examples:
        "> 50%"          → (">",  50.0,     "%")
        "≥ 100,000/mm³"  → (">=", 100000.0, "/mm³")
        "≤ 2.2 mg/dL"   → ("<=", 2.2,      "mg/dL")
        "≥ 18 years"     → (">=", 18.0,     "years")

    Returns None if the string cannot be parsed.
    """
    m = _THRESHOLD_RE.search(s)
    if not m:
        return None
    op_raw = m.group(1)
    val_str = m.group(2)
    unit = m.group(3).strip()
    op = _OP_MAP.get(op_raw, op_raw)
    try:
        value = float(val_str.replace(",", ""))
    except ValueError:
        return None
    return op, value, unit


def _compare(patient_val: float, op: str, threshold_val: float) -> bool:
    if op == ">=":
        return patient_val >= threshold_val
    if op == ">":
        return patient_val > threshold_val
    if op == "<=":
        return patient_val <= threshold_val
    if op == "<":
        return patient_val < threshold_val
    if op == "=":
        return patient_val == threshold_val
    return False


def _first_threshold(thresholds: list[str]) -> tuple[str, float, str] | None:
    """Return the first successfully parsed threshold from the list."""
    for t in thresholds:
        parsed = _parse_threshold(t)
        if parsed:
            return parsed
    return None


def _threshold_from_text(text: str) -> tuple[str, float, str] | None:
    """Fallback: scan criterion text directly for a threshold expression."""
    return _parse_threshold(text)


# ---------------------------------------------------------------------------
# Keyword routing — map criterion text to patient profile fields
# ---------------------------------------------------------------------------

_SEX_FEMALE_RE = re.compile(r"\b(female|women|woman)\b", re.I)
_SEX_MALE_RE   = re.compile(r"\b(male|men|man)\b", re.I)

_LAB_KEYWORDS: dict[str, re.Pattern] = {
    "platelet_count":   re.compile(r"\bplatelet\b", re.I),
    "hemoglobin":       re.compile(r"\bhemoglobin\b|\bHgb\b|\b\bHb\b", re.I),
    "neutrophil_count": re.compile(r"\bneutrophil\b|\bANC\b|\bgranulocyte\b", re.I),
    "creatinine":       re.compile(r"\bcreatinine\b", re.I),
    "bilirubin":        re.compile(r"\bbilirubin\b", re.I),
    "alt":              re.compile(r"\bALT\b|\balanine aminotransferase\b", re.I),
    "ast":              re.compile(r"\bAST\b|\baspartate aminotransferase\b", re.I),
    "lvef":             re.compile(
        r"\bLVEF\b|\bleft ventricular ejection fraction\b|\bejection fraction\b",
        re.I,
    ),
    "testosterone":     re.compile(r"\btestosterone\b", re.I),
}


def evaluate_objective_criterion(
    criterion: Criterion,
    patient: dict,
) -> bool | None:
    """
    Compare a patient profile against a single objective criterion.

    Routing order: sex → ECOG → Karnofsky → age → lab values.
    Returns None when the criterion cannot be matched to any patient field.

    Returns:
        True  — patient is eligible w.r.t. this criterion
        False — patient is NOT eligible w.r.t. this criterion (hard disqualifier)
        None  — cannot determine from available patient profile fields

    For exclusion criteria (b1_label=0), "eligible w.r.t." means the patient
    does NOT trigger the exclusion. The inversion is applied here so callers
    always receive a uniform "is eligible" bool.
    """
    text = criterion.text
    thresholds = criterion.extracted_thresholds or []
    is_exclusion = (criterion.b1_label == 0)

    # -- Sex ---------------------------------------------------------------
    has_female = bool(_SEX_FEMALE_RE.search(text))
    has_male   = bool(_SEX_MALE_RE.search(text))
    if has_female or has_male:
        patient_sex = (patient.get("sex") or "").lower()
        if patient_sex:
            if has_female and not has_male:
                required = "female"
            elif has_male and not has_female:
                required = "male"
            else:
                required = None  # mentions both — ambiguous
            if required:
                meets = patient_sex == required
                return (not meets) if is_exclusion else meets

    # -- ECOG --------------------------------------------------------------
    # Narrow to explicit ECOG mention only — "performance status" alone also
    # appears in Karnofsky criteria and must not be caught here first.
    if re.search(r"\bECOG\b", text, re.I):
        ecog = patient.get("ecog")
        if ecog is not None:
            parsed = _first_threshold(thresholds) or _threshold_from_text(text)
            if parsed:
                op, val, _ = parsed
                meets = _compare(float(ecog), op, val)
                return (not meets) if is_exclusion else meets

    # -- Karnofsky ---------------------------------------------------------
    if re.search(r"\bkarnofsky\b", text, re.I):
        kps = patient.get("karnofsky")
        if kps is not None:
            parsed = _first_threshold(thresholds) or _threshold_from_text(text)
            if parsed:
                op, val, _ = parsed
                meets = _compare(float(kps), op, val)
                return (not meets) if is_exclusion else meets

    # -- Age ---------------------------------------------------------------
    if re.search(r"\bage\b|\byears? old\b|\byears? of age\b", text, re.I):
        age = patient.get("age")
        if age is not None:
            # Prefer threshold that mentions "year" in its unit
            parsed = None
            for t in thresholds:
                p = _parse_threshold(t)
                if p and re.search(r"year", p[2], re.I):
                    parsed = p
                    break
            if not parsed:
                parsed = _first_threshold(thresholds) or _threshold_from_text(text)
            if parsed:
                op, val, _ = parsed
                meets = _compare(float(age), op, val)
                return (not meets) if is_exclusion else meets

    # -- Lab values --------------------------------------------------------
    lab_values = patient.get("lab_values") or {}
    for field_name, pattern in _LAB_KEYWORDS.items():
        if pattern.search(text) and field_name in lab_values:
            parsed = _first_threshold(thresholds) or _threshold_from_text(text)
            if parsed:
                op, val, _ = parsed
                meets = _compare(float(lab_values[field_name]), op, val)
                return (not meets) if is_exclusion else meets

    return None  # No matching patient field found


# ---------------------------------------------------------------------------
# Hedging estimation for subjective criteria
# ---------------------------------------------------------------------------

_HEDGING_HIGH_RE = re.compile(
    r"\b(willing|willingness|ability to|adequate judgment|acceptable to|"
    r"in the opinion of|physician|investigator|clinician|life expectancy|"
    r"geographic|travel|comply|compliance|adequate)\b",
    re.I,
)
_HEDGING_LOW_RE = re.compile(
    r"\b(signed|written consent|informed consent|IRB|institutional review)\b",
    re.I,
)


def estimate_hedging(text: str) -> float:
    """
    Heuristic hedging strength for a subjective criterion [0.0, 1.0].

    Used by eligibility_model.py to shape the Beta prior:
        alpha = 2.0 * (1 - hedging)
        beta  = 2.0 * hedging

    High hedging (0.8): physician judgment, willingness, life expectancy —
        genuinely uncertain → Beta(0.4, 1.6) skewed toward uncertain.
    Low hedging (0.05): written consent — nearly always met in practice →
        Beta(1.9, 0.1) skewed strongly toward met.
    Default (0.5): maximum uncertainty → Beta(1.0, 1.0) uninformative.
    """
    if _HEDGING_HIGH_RE.search(text):
        return 0.8
    if _HEDGING_LOW_RE.search(text):
        return 0.05
    return 0.5


# ---------------------------------------------------------------------------
# DuckDB loader
# ---------------------------------------------------------------------------

def load_criteria_for_trial(nct_id: str, con) -> list[Criterion]:
    """
    Fetch all criteria for a trial from DuckDB and return as Criterion objects.

    NULL B2/B3 labels are preserved — the Bayesian model marginalizes over
    criteria whose labels cannot be determined.

    Args:
        nct_id: trial identifier (e.g. "NCT00127920")
        con:    active DuckDB connection

    Returns:
        list of Criterion objects ordered by position within the trial
    """
    rows = con.execute(
        """
        SELECT
            criterion_id,
            text,
            b1_label,
            b2_label,
            b3_label,
            COALESCE(b2_confidence, 0.0)   AS b2_confidence,
            COALESCE(b3_confidence, 0.0)   AS b3_confidence,
            COALESCE(extracted_thresholds,   []) AS extracted_thresholds,
            COALESCE(extracted_demographics, []) AS extracted_demographics,
            COALESCE(extracted_lab_values,   []) AS extracted_lab_values,
            COALESCE(extracted_conditions,   []) AS extracted_conditions,
            COALESCE(extracted_drugs,        []) AS extracted_drugs,
            COALESCE(extracted_scales,       []) AS extracted_scales
        FROM criteria
        WHERE nct_id = ?
        ORDER BY position
        """,
        [nct_id],
    ).fetchall()

    return [
        Criterion(
            criterion_id=row[0],
            text=row[1] or "",
            b1_label=row[2],
            b2_label=row[3],
            b3_label=row[4],
            b2_confidence=float(row[5]),
            b3_confidence=float(row[6]),
            extracted_thresholds=list(row[7]),
            extracted_demographics=list(row[8]),
            extracted_lab_values=list(row[9]),
            extracted_conditions=list(row[10]),
            extracted_drugs=list(row[11]),
            extracted_scales=list(row[12]),
        )
        for row in rows
    ]
