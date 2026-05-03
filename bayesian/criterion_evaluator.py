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
    prior_chemo   (bool) — True if patient received prior chemotherapy
    prior_rt      (bool) — True if patient received prior radiation therapy
    brain_mets    (bool) — True if patient has active brain metastases
    pregnant      (bool) — True if patient is pregnant or breastfeeding
                           (omit/None → pregnancy criteria become UNOBSERVABLE)
    nyha_class    (int: 1–4) — NYHA cardiac functional classification
    child_pugh    (str: "A" | "B" | "C") — Child-Pugh hepatic class
    lab_values    (dict[str, float]):
        platelet_count    — /mm³
        hemoglobin        — g/dL
        neutrophil_count  — /mm³ (ANC / granulocyte count)
        wbc               — /mm³ (total white blood cell count)
        inr               — ratio
        aptt              — seconds
        creatinine        — mg/dL (serum; for GFR/clearance criteria use egfr)
        egfr              — mL/min/1.73m² (also handles creatinine clearance criteria)
        bilirubin         — mg/dL
        alt               — U/L
        ast               — U/L
        albumin           — g/dL
        lvef              — %
        qtc               — ms
        calcium           — mg/dL
        glucose           — mg/dL
        potassium         — mEq/L
        ldh               — U/L
        psa               — ng/mL
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
    # Longer phrases first — prevents "greater than" from matching inside
    # "no greater than" before the compound phrase rule fires.
    (re.compile(r'\bgreater\s+than\s+or\s+equal\s+to\b', re.I), '>='),
    (re.compile(r'\bless\s+than\s+or\s+equal\s+to\b',    re.I), '<='),
    (re.compile(r'\bno\s+(?:more|greater)\s+than\b',      re.I), '<='),
    (re.compile(r'\bnot\s+(?:more|greater)\s+than\b',     re.I), '<='),
    (re.compile(r'\bno\s+less\s+than\b',                  re.I), '>='),
    (re.compile(r'\bgreater\s+than\b',                    re.I), '>'),
    (re.compile(r'\bless\s+than\b',                       re.I), '<'),
    (re.compile(r'\bat\s+least\b',                        re.I), '>='),
    (re.compile(r'\bat\s+most\b',                         re.I), '<='),
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
    # Haematology
    "platelet_count":   re.compile(r"\bplatelet\b", re.I),
    "hemoglobin":       re.compile(r"\bhemoglobin\b|\bHgb\b|\bHb\b", re.I),
    "neutrophil_count": re.compile(r"\bneutrophil\b|\bANC\b|\bgranulocyte\b", re.I),
    "wbc":              re.compile(r"\bWBC\b|\bwhite\s+blood\s+cell(?:\s+count)?\b|\bleukocyte\s+count\b", re.I),
    # Coagulation
    "inr":              re.compile(r"\bINR\b|\binternational\s+normalized\s+ratio\b", re.I),
    "aptt":             re.compile(r"\baPTT\b|\bAPTT\b|\bactivated\s+partial\s+thromboplastin\b", re.I),
    # Renal — creatinine = serum (mg/dL); egfr also catches creatinine clearance (mL/min)
    "creatinine":       re.compile(r"\bcreatinine\b", re.I),
    "egfr":             re.compile(r"\beGFR\b|\bGFR\b|\bglomerular\s+filtration\s+rate\b|\bcreatinine\s+clearance\b", re.I),
    # Hepatic
    "bilirubin":        re.compile(r"\bbilirubin\b", re.I),
    "alt":              re.compile(r"\bALT\b|\balanine\s+aminotransferase\b", re.I),
    "ast":              re.compile(r"\bAST\b|\baspartate\s+aminotransferase\b", re.I),
    "albumin":          re.compile(r"\balbumin\b", re.I),
    # Cardiac
    "lvef":             re.compile(r"\bLVEF\b|\bleft\s+ventricular\s+ejection\s+fraction\b|\bejection\s+fraction\b", re.I),
    "qtc":              re.compile(r"\bQTc\b|\bQTcF\b|\bQTcB\b|\bQT\s+(?:interval|corrected|prolongation)\b", re.I),
    # Metabolic / Chemistry
    "calcium":          re.compile(r"\bcalcium\b", re.I),
    "glucose":          re.compile(r"\bglucose\b|\bblood\s+sugar\b", re.I),
    "potassium":        re.compile(r"\bpotassium\b", re.I),
    "ldh":              re.compile(r"\bLDH\b|\blactate\s+dehydrogenase\b", re.I),
    # Tumour markers / reproductive
    "psa":              re.compile(r"\bPSA\b|\bprostate[- ]specific\s+antigen\b", re.I),
    "testosterone":     re.compile(r"\btestosterone\b", re.I),
}

# ---------------------------------------------------------------------------
# ULN multiplier guard — skip lab criteria expressed as multiples of ULN
# ---------------------------------------------------------------------------
#
# Criteria like "AST ≤ 2.5 × ULN" express the threshold as a small multiplier
# (2.5) of the institution-specific upper limit of normal, not as an absolute
# lab value. Without knowing the ULN we cannot safely compare a patient value
# (AST = 28 U/L) against a multiplier (2.5). The threshold parser would extract
# 2.5 and produce a hard fail (28 > 2.5), which is a false positive.
#
# Fix: detect ULN-multiplier language and return None → UNOBSERVABLE.
# Only absolute-value criteria ("Bilirubin ≤ 1.5 mg/dL") continue to the
# numeric comparison; ULN expressions are intentionally unresolved.

_ULN_MULTIPLIER_RE = re.compile(
    r"\b(?:times?|×|x)\s*(?:(?:upper\s+)?(?:limit\s+of\s+)?normal|ULN)\b"
    r"|\bULN\b"
    r"|\btimes\s+normal\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Boolean / categorical clinical field routing
# ---------------------------------------------------------------------------

_PRIOR_CHEMO_MENTION_RE = re.compile(r"\bchemo(?:therapy)?\b", re.I)
_PRIOR_RT_MENTION_RE    = re.compile(
    r"\bradiation\s+(?:therapy|treatment)\b|\bradiotherapy\b", re.I
)

# Concurrent-treatment guard — criteria about *currently* administered therapy
# are distinct from prior therapy history. The patient profile records prior
# chemo/RT (bool) but not concurrent/active treatment status. Routing a
# "no concurrent chemotherapy" exclusion through prior_chemo would produce a
# false fail for any patient with a prior_chemo=True history.
# Guard: if "concurrent" appears in the criterion text, skip therapy routing.
_CONCURRENT_RE = re.compile(r"\bconcurrent\b", re.I)
# Pregnancy / lactation exclusion — must only fire when patient explicitly
# states pregnant=True. If the field is absent, return None (→ UNOBSERVABLE).
_PREGNANCY_RE = re.compile(
    r"\bpregnant\b|\bpregnancy\b|\blactati(?:ng|on)\b|\bbreastfeed(?:ing)?\b|"
    r"\bnursing\b|\bbreast[\s-]?feed(?:ing)?\b|\bchild[-\s]?bearing\b",
    re.I,
)

_BRAIN_METS_RE = re.compile(
    r"\bbrain\s+(?:metasta[sz]\w*|involvement|tumor|lesion|disease)\b|"
    r"\bCNS\s+metasta[sz]\w*\b|\bintracranial\s+metasta[sz]\w*\b",
    re.I,
)

# Exclusion direction: criterion requires patient to NOT have had prior therapy
_EXCL_THERAPY_DIRECTION_RE = re.compile(
    r"\bno\s+prior\b|\bwithout\s+prior\b|\bna[ïi]ve\b|\buntreated\b|"
    r"\bineligible\b|\bnot\s+(?:allowed|permitted|eligible)\b|"
    r"\bprohibit\b|\bmakes?\s+(?:a\s+patient\s+)?ineligible\b|"
    r"\bpreviously\s+untreated\b|\btreatment[- ]na[ïi]ve\b|"
    r"\bchemo(?:therapy)?[- ]na[ïi]ve\b",
    re.I,
)
# Inclusion direction: criterion requires patient to have had prior therapy
_INCL_THERAPY_DIRECTION_RE = re.compile(
    r"\bat\s+least\s+(?:one|1|two|2|\d+)\s+(?:prior|previous)\b|"
    r"\bpreviously\s+treated\b|\breceived\s+(?:prior|previous)\b|"
    r"\bone\s+or\s+more\s+prior\b|\b\d+\s+prior\s+(?:line|regimen|course)\b",
    re.I,
)


def _normalize_roman_numerals(text: str) -> str:
    """Replace NYHA Roman numeral class labels I–IV with integers.

    Processes in reverse length order (IV before I) to avoid partial matches.
    Standalone 'I' is only converted in the context of 'class I' to avoid
    false matches in ordinary English text.
    """
    text = re.sub(r"\bIV\b", "4", text)
    text = re.sub(r"\bIII\b", "3", text)
    text = re.sub(r"\bII\b", "2", text)
    text = re.sub(r"\bclass\s+I\b", "class 1", text, flags=re.I)
    return text


def evaluate_objective_criterion(
    criterion: Criterion,
    patient: dict,
) -> bool | None:
    """
    Compare a patient profile against a single objective criterion.

    Routing order: sex → pregnancy → ECOG → Karnofsky → prior therapy →
                   brain mets → NYHA → Child-Pugh → age → lab values.
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

    # -- Compound criterion guard ------------------------------------------
    # Compound criteria pack multiple distinct fields into one text block
    # (e.g. "WBC ≥ 3000, Platelet ≥ 100,000, Hgb ≥ 8.0, Bilirubin ≤ 1.5...").
    # The keyword-first routing picks the first matching field and the first
    # threshold in the whole block, which may belong to a different field —
    # producing incorrect hard pass/fail results. Guard: if three or more
    # distinct lab keyword patterns match the text, the criterion is too
    # compound to evaluate safely; return None → UNOBSERVABLE.
    _lab_hits = sum(1 for pat in _LAB_KEYWORDS.values() if pat.search(text))
    if _lab_hits >= 3:
        return None

    # -- Pregnancy / lactation ---------------------------------------------
    # Only evaluate when patient explicitly provides a `pregnant` boolean.
    # If the field is absent, return None so the criterion becomes UNOBSERVABLE
    # rather than a spurious hard disqualifier for every female patient.
    if _PREGNANCY_RE.search(text):
        pregnant = patient.get("pregnant")
        if pregnant is not None:
            # Pregnancy criteria are universally exclusions; patient is eligible
            # w.r.t. this criterion only if they are not pregnant/lactating.
            return not bool(pregnant)
        # Field absent — cannot evaluate; fall through to return None below
        return None

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

    # -- Prior chemotherapy ------------------------------------------------
    # Direction detection order:
    #   1. Explicit exclusion-direction phrases ("no prior", "naive", etc.)
    #      OR criterion is labeled EXC (b1_label=0) → patient must be chemo-naive.
    #   2. Explicit inclusion-direction phrases ("at least 1 prior line", etc.)
    #      → patient must have had chemo.
    #   3. Fallback to b1_label: INC (b1_label=1) → patient must have had chemo;
    #      b1_label=None → cannot determine, return None.
    if _PRIOR_CHEMO_MENTION_RE.search(text):
        # Concurrent-treatment criteria cannot be evaluated from prior_chemo;
        # returning None sends them to UNOBSERVABLE rather than a false fail.
        if _CONCURRENT_RE.search(text):
            return None
        prior_chemo = patient.get("prior_chemo")
        if prior_chemo is not None:
            if _EXCL_THERAPY_DIRECTION_RE.search(text) or is_exclusion:
                return not bool(prior_chemo)
            if _INCL_THERAPY_DIRECTION_RE.search(text):
                return bool(prior_chemo)
            # Fallback: trust b1_label when text direction is ambiguous
            if criterion.b1_label == 1:
                return bool(prior_chemo)
            # b1_label=None — too ambiguous to call
            # fall through to return None

    # -- Prior radiation therapy -------------------------------------------
    if _PRIOR_RT_MENTION_RE.search(text):
        if _CONCURRENT_RE.search(text):
            return None
        prior_rt = patient.get("prior_rt")
        if prior_rt is not None:
            if _EXCL_THERAPY_DIRECTION_RE.search(text) or is_exclusion:
                return not bool(prior_rt)
            if _INCL_THERAPY_DIRECTION_RE.search(text):
                return bool(prior_rt)
            if criterion.b1_label == 1:
                return bool(prior_rt)

    # -- Brain metastases --------------------------------------------------
    # Almost universally an exclusion criterion. Rare brain-tumour inclusion
    # trials are not handled to avoid false positives.
    if _BRAIN_METS_RE.search(text):
        brain_mets = patient.get("brain_mets")
        if brain_mets is not None:
            return not bool(brain_mets)

    # -- NYHA class --------------------------------------------------------
    if re.search(r"\bNYHA\b", text, re.I):
        nyha = patient.get("nyha_class")
        if nyha is not None:
            norm = _normalize_roman_numerals(text)
            parsed = _first_threshold(thresholds) or _threshold_from_text(norm)
            if parsed:
                op, val, _ = parsed
                meets = _compare(float(nyha), op, val)
                return (not meets) if is_exclusion else meets
            # Fallback: list of class numbers mentioned → treat max as ≤ threshold
            nums = [int(m) for m in re.findall(r"\b([1-4])\b", norm)]
            if nums:
                meets = float(nyha) <= max(nums)
                return (not meets) if is_exclusion else meets

    # -- Child-Pugh class --------------------------------------------------
    if re.search(r"\bChild[-\s]?Pugh\b", text, re.I):
        cp = patient.get("child_pugh")
        if cp:
            cp_val = {"A": 1, "B": 2, "C": 3}.get(str(cp).upper())
            if cp_val is not None:
                classes_found = re.findall(r"\b([ABC])\b", text, re.I)
                if classes_found:
                    max_allowed = max(
                        {"A": 1, "B": 2, "C": 3}.get(c.upper(), 0)
                        for c in classes_found
                    )
                    if max_allowed > 0:
                        meets = cp_val <= max_allowed
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
            # ULN-multiplier guard: threshold is expressed as a multiple of the
            # upper limit of normal (e.g. "≤ 2.5 × ULN"). Parsing extracts 2.5
            # as an absolute value, which would incorrectly compare AST=28 against
            # a threshold of 2.5 and produce a false hard fail. We cannot safely
            # evaluate without knowing the institution-specific ULN; return None.
            if _ULN_MULTIPLIER_RE.search(text):
                return None
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
    Default (0.35): moderately optimistic → Beta(1.3, 0.7) mean ≈ 0.65.
        Reflects the trial-seeking population selection effect: patients
        who reach the eligibility screening step typically meet most
        protocol requirements that cannot be objectively assessed from
        structured data. Lowered from 0.5 (maximum uncertainty / Beta(1,1))
        to reduce multiplicative shrinkage when several subjective criteria
        appear together.
    """
    if _HEDGING_HIGH_RE.search(text):
        return 0.8
    if _HEDGING_LOW_RE.search(text):
        return 0.05
    return 0.35


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
