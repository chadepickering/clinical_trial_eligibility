"""
Regex/heuristic weak supervision for bootstrap labeling.

Generates noisy labels for B1/B2/B3 classification tasks to support
training before manual annotation is available.

B1: Inclusion (1) vs Exclusion (0) — derived from section context
    Confidence ~0.90 — reliable except for embedded exclusions in inclusion
    sections (e.g. "were excluded" phrasing) which the splitter cannot resolve.

B2: Objective (1) vs Subjective (0) — regex pattern matching
    Confidence ~0.75 — numeric thresholds and named scales are strong signals;
    subjective language ("adequate", "willing") is clear but overlaps exist.

B3: Observable (1) vs Unobservable (0) — EHR field lookup + signal keywords
    Confidence ~0.70 — standard lab/demographic fields are reliable; history
    and consent language is reliable; many criteria are genuinely ambiguous.

All labels are None when evidence is absent or contradictory.
"""

import re


# ---------------------------------------------------------------------------
# B2 patterns — Objective vs Subjective
# ---------------------------------------------------------------------------

# Objective: numeric thresholds, named clinical scales, units of measurement
_OBJECTIVE_PATTERNS = [
    re.compile(r'\d+\.?\d*\s*(%|mg|ml|mcl|mmol|mmhg|ng|g/dl|g/l|u/l|iu/l|mm\^3|mm3|mL/min)', re.IGNORECASE),
    re.compile(r'[≥≤<>]=?\s*\d'),                          # comparator + number e.g. "≥ 18", "> 2 cm"
    re.compile(r'\d+\s*(years?|days?|weeks?|months?)\s*(of age|old)?', re.IGNORECASE),
    re.compile(r'(ECOG|NYHA|CTCAE|WHO|Karnofsky|KPS|Zubrod|IPSS|NRS|PHQ|ISI)\s*[\w\s\-]*\d', re.IGNORECASE),
    re.compile(r'(performance status|grade|class|stage|score)\s*[≤≥<>=\-–]\s*\d', re.IGNORECASE),
    re.compile(r'(platelet|neutrophil|lymphocyte|hemoglobin|creatinine|bilirubin|egfr|gfr|ast|alt|wbc|psa|hba1c|inr|aptt|ptt)\s*[\w\s]*[≤≥<>]', re.IGNORECASE),
    re.compile(r'x\s*(upper limit of normal|ULN)', re.IGNORECASE),
    re.compile(r'(bmi|body mass index)\s*[≤≥<>]\s*\d', re.IGNORECASE),
    re.compile(r'\b(normal|within normal limits|per institutional|normal range)\b', re.IGNORECASE),
]

# Subjective: judgment-dependent language, consent, willingness
_SUBJECTIVE_PATTERNS = [
    re.compile(r'\b(adequate|sufficient|significant|clinically\s+significant|appropriate|reasonable|acceptable)\b', re.IGNORECASE),
    re.compile(r'\b(life expectancy|prognosis|functional status|performance)\b(?!.*\d)', re.IGNORECASE),
    re.compile(r'\b(willing|able|capable|agrees?\s+to|consent|assent)\b', re.IGNORECASE),
    re.compile(r'\bat the (discretion|judgment|judgement) of\b', re.IGNORECASE),
    re.compile(r'\b(well-controlled|uncontrolled|poorly controlled)\b(?!.*\d)', re.IGNORECASE),
    re.compile(r'\b(stable|unstable)\b(?!.*\d)', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# B3 patterns — Observable vs Unobservable
# ---------------------------------------------------------------------------

# Observable: standard EHR fields — lab values, demographics, diagnoses, vitals
_OBSERVABLE_PATTERNS = [
    # Lab values — extended to cover additional common oncology labs
    re.compile(r'\b(hba1c|egfr|gfr|creatinine|hemoglobin|haemoglobin|platelet|wbc|alt|ast|bilirubin|albumin|ldh|inr|aptt|ptt|psa|cea|ca125|ferritin|sodium|potassium|calcium|magnesium|phosphate|glucose|lipase|amylase|troponin|uric acid|fibrinogen|hematocrit|haematocrit|neutrophil|lymphocyte|eosinophil|monocyte)\b', re.IGNORECASE),
    # Cardiac imaging and functional measures
    re.compile(r'\b(ejection fraction|lvef|left ventricular|echocardiograph|echo|lvef|qt|qtc|ecg|electrocardiogram)\b', re.IGNORECASE),
    # Vitals and anthropometrics
    re.compile(r'\b(age|bmi|body mass index|weight|height|blood pressure|systolic|diastolic|heart rate|temperature|oxygen saturation|spo2)\b', re.IGNORECASE),
    # Diagnosis, staging, and disease state
    re.compile(r'\b(diagnosis|histolog|cytolog|patholog|biopsy|stage|grade|tumor|tumour|malignancy|cancer|carcinoma|lymphoma|leukemia|leukaemia|metastas\w+|metastat\w+|refractory|relapsed|progressive|resistant)\b', re.IGNORECASE),
    # Performance scales (observable via clinical assessment)
    re.compile(r'\b(ECOG|Karnofsky|KPS|NYHA|WHO performance|Zubrod)\b', re.IGNORECASE),
    # Medications and treatment history (in EHR)
    re.compile(r'\b(prior|previous|received|treated with|treatment with|therapy with|regimen|chemotherapy|immunotherapy|radiotherapy|radiation)\b', re.IGNORECASE),
    # Reproductive status and pregnancy (documented via test or record)
    re.compile(r'\b(pregnant|pregnancy test|breastfeeding|breast.feeding|post.menopausal|pre.menopausal|menarche|childbearing)\b', re.IGNORECASE),
    # Known history — documented in records
    re.compile(r'\b(known|documented|confirmed|history of|evidence of|diagnosis of)\b', re.IGNORECASE),
    # Hypersensitivity and allergy — documented in medical history
    re.compile(r'\b(hypersensitivity|sensitivity|allerg\w+)\b', re.IGNORECASE),
]

# Unobservable: consent, intent, access, life expectancy, geographic constraints
_UNOBSERVABLE_PATTERNS = [
    re.compile(r'\b(willing|consent|assent|agrees?\s+to|agrees?\s+not\s+to)\b', re.IGNORECASE),
    re.compile(r'\b(life expectancy|expected survival|estimated survival)\b', re.IGNORECASE),
    re.compile(r'\b(access to|able to (attend|travel|use|read|write|understand|communicate))\b', re.IGNORECASE),
    re.compile(r'\b(geographic|travel|transportation|distance)\b', re.IGNORECASE),
    re.compile(r'\b(investigator.{0,15}(discretion|judgment|opinion))\b', re.IGNORECASE),
    re.compile(r'\b(planning to|intend(s|ing)? to|expecting to|anticipate)\b', re.IGNORECASE),
    re.compile(r'\b(internet|phone|telephone|device|technology)\b', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def _count_matches(text: str, patterns: list) -> int:
    return sum(1 for p in patterns if p.search(text))


def label_criterion(criterion: dict) -> dict:
    """
    Returns a copy of the criterion dict augmented with weak label fields:

        b1_label        int | None   — 1=inclusion, 0=exclusion, None=unknown
        b2_label        int | None   — 1=objective, 0=subjective, None=ambiguous
        b3_label        int | None   — 1=observable, 0=unobservable, None=ambiguous
        b2_confidence   float        — 0.0–1.0 signal strength for B2
        b3_confidence   float        — 0.0–1.0 signal strength for B3
    """
    text = criterion.get('text', '')
    section = criterion.get('section', 'unknown')

    result = dict(criterion)

    # --- B1: derived directly from section context ---
    if section == 'inclusion':
        result['b1_label'] = 1
    elif section == 'exclusion':
        result['b1_label'] = 0
    else:
        result['b1_label'] = None

    # --- B2: objective vs subjective ---
    obj_hits = _count_matches(text, _OBJECTIVE_PATTERNS)
    subj_hits = _count_matches(text, _SUBJECTIVE_PATTERNS)
    total_b2 = obj_hits + subj_hits

    if total_b2 == 0:
        result['b2_label'] = None
        result['b2_confidence'] = 0.0
    elif obj_hits > subj_hits:
        result['b2_label'] = 1  # objective
        result['b2_confidence'] = round(obj_hits / total_b2, 2)
    elif subj_hits > obj_hits:
        result['b2_label'] = 0  # subjective
        result['b2_confidence'] = round(subj_hits / total_b2, 2)
    else:
        # Equal hits — ambiguous
        result['b2_label'] = None
        result['b2_confidence'] = 0.5

    # --- B3: observable vs unobservable ---
    obs_hits = _count_matches(text, _OBSERVABLE_PATTERNS)
    unobs_hits = _count_matches(text, _UNOBSERVABLE_PATTERNS)
    total_b3 = obs_hits + unobs_hits

    if total_b3 == 0:
        result['b3_label'] = None
        result['b3_confidence'] = 0.0
    elif obs_hits > unobs_hits:
        result['b3_label'] = 1  # observable
        result['b3_confidence'] = round(obs_hits / total_b3, 2)
    elif unobs_hits > obs_hits:
        result['b3_label'] = 0  # unobservable
        result['b3_confidence'] = round(unobs_hits / total_b3, 2)
    else:
        result['b3_label'] = None
        result['b3_confidence'] = 0.5

    return result
