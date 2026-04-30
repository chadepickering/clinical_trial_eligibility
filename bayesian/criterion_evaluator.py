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


# English-language comparator phrases → symbolic operators.
# Applied by _threshold_from_text before the regex runs so that criteria
# written in prose ("less than or equal to ECOG 2") are handled the same
# way as symbolic ones ("ECOG ≤ 2").
_ENGLISH_OP_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bgreater\s+than\s+or\s+equal\s+to\b', re.I), '>='),
    (re.compile(r'\bless\s+than\s+or\s+equal\s+to\b',    re.I), '<='),
    (re.compile(r'\bgreater\s+than\b',                    re.I), '>'),
    (re.compile(r'\bless\s+than\b',                       re.I), '<'),
    (re.compile(r'\bat\s+least\b',                        re.I), '>='),
    (re.compile(r'\bat\s+most\b',                         re.I), '<='),
    (re.compile(r'\bno\s+more\s+than\b',                  re.I), '<='),
    (re.compile(r'\bno\s+less\s+than\b',                  re.I), '>='),
    (re.compile(r'\bnot\s+exceed(?:ing)?\b',              re.I), '<='),
]

# Loose threshold: operator followed by optional non-numeric words then a number.
# Handles "≤ ECOG 2" or "> grade 1" after English normalization.
_LOOSE_THRESHOLD_RE = re.compile(
    r'(>=|<=|>|<)\s*(?:[A-Za-z]+\s+){0,3}(\d[\d,]*\.?\d*)',
)


def _normalize_english_ops(text: str) -> str:
    """Replace English comparison phrases with symbolic operators."""
    for pattern, symbol in _ENGLISH_OP_SUBS:
        text = pattern.sub(symbol, text)
    return text


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
    """
    Scan criterion text for a threshold expression.

    Three-pass strategy:
      1. Strict symbolic parse on the original text.
      2. Normalize English comparators ("less than or equal to" → "<="),
         then strict parse again.
      3. Loose parse: operator followed by optional intervening words then
         a number (e.g. "≤ ECOG 2" or "> grade 1").
    """
    # Pass 1: original text, strict
    result = _parse_threshold(text)
    if result:
        return result

    # Pass 2: normalize English operators, then strict
    normalized = _normalize_english_ops(text)
    result = _parse_threshold(normalized)
    if result:
        return result

    # Pass 3: loose — operator + up to 3 optional words + number
    m = _LOOSE_THRESHOLD_RE.search(normalized)
    if m:
        op = m.group(1)
        try:
            value = float(m.group(2).replace(',', ''))
        except ValueError:
            return None
        return op, value, ''

    return None


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
            # Creatinine clearance (cc/min or mL/min) is a different measurement
            # from serum creatinine (mg/dL). The patient profile stores serum
            # creatinine; skip criteria that explicitly reference clearance to
            # avoid comparing 0.9 mg/dL against a 60 cc/min threshold.
            if field_name == "creatinine" and re.search(
                r'\bclearance\b', text, re.I
            ):
                continue
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
# Synthetic criteria from trial-level metadata
# ---------------------------------------------------------------------------

_AGE_DIGITS_RE = re.compile(r"(\d+)")


def _parse_age_years(age_str: str | None) -> int | None:
    """Parse '18 Years' → 18, None / '' → None."""
    if not age_str:
        return None
    m = _AGE_DIGITS_RE.search(age_str)
    return int(m.group(1)) if m else None


def _synthetic_criteria_from_metadata(nct_id: str, con) -> list[Criterion]:
    """
    Build Criterion objects from trial-level metadata fields that are not
    captured in the NLP-split criteria text:

        trials.sex     → sex eligibility criterion (FEMALE / MALE only;
                          ALL and None produce no criterion)
        trials.min_age → minimum age inclusion criterion
        trials.max_age → maximum age inclusion criterion (uncommon; mostly NULL)

    All synthetic criteria are labeled B2=1 (objective) + B3=1 (observable)
    so they enter the deterministic branch of evaluate_all_criteria.

    Criterion IDs carry a ``_meta_`` infix so they are distinguishable from
    NLP-split criteria in downstream output and tests.
    """
    row = con.execute(
        "SELECT sex, min_age, max_age FROM trials WHERE nct_id = ?", [nct_id]
    ).fetchone()
    if not row:
        return []

    sex, min_age_str, max_age_str = row
    synthetic: list[Criterion] = []

    # -- Sex -----------------------------------------------------------------
    if sex and sex.upper() in ("FEMALE", "MALE"):
        label = "Female" if sex.upper() == "FEMALE" else "Male"
        synthetic.append(
            Criterion(
                criterion_id=f"{nct_id}_meta_sex",
                text=f"{label} patients only",
                b1_label=1,
                b2_label=1,
                b3_label=1,
            )
        )

    # -- Minimum age ---------------------------------------------------------
    min_years = _parse_age_years(min_age_str)
    if min_years is not None:
        synthetic.append(
            Criterion(
                criterion_id=f"{nct_id}_meta_min_age",
                text=f"Age ≥ {min_years} years",
                b1_label=1,
                b2_label=1,
                b3_label=1,
                extracted_thresholds=[f"≥ {min_years} years"],
            )
        )

    # -- Maximum age ---------------------------------------------------------
    max_years = _parse_age_years(max_age_str)
    if max_years is not None:
        synthetic.append(
            Criterion(
                criterion_id=f"{nct_id}_meta_max_age",
                text=f"Age ≤ {max_years} years",
                b1_label=1,
                b2_label=1,
                b3_label=1,
                extracted_thresholds=[f"≤ {max_years} years"],
            )
        )

    return synthetic


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
        list of Criterion objects ordered by position within the trial,
        followed by any synthetic criteria derived from trial-level metadata
        (sex eligibility, min/max age) that are not present in the NLP-split
        criteria text.
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
    ] + _synthetic_criteria_from_metadata(nct_id, con)
