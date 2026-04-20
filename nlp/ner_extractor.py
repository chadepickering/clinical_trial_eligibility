"""
Named entity extraction for clinical trial eligibility criteria.

Extracts seven entity types per criterion and writes them to the criteria table:
    extracted_conditions    — cancer types and comorbidities
    extracted_drugs         — drug/intervention names and aliases
    extracted_lab_values    — lab test names (HbA1c, eGFR, etc.)
    extracted_thresholds    — numeric thresholds with comparators and units
    extracted_demographics  — age, sex, BMI patterns
    extracted_scales        — named clinical scales with scores
    extracted_timeframes    — time windows ("within 6 months")

Strategy by entity type:
    CONDITION / DRUG  — trial-specific dictionary match using MeSH terms and
                        drug aliases already in the trials table. More reliable
                        than general NER for in-domain entities; MeSH provides
                        normalised labels.
    LAB_VALUE         — regex over curated lab name list
    THRESHOLD         — regex: comparator + number + unit
    SCALE             — regex: named scale + optional score
    DEMOGRAPHIC       — regex: age/sex/BMI patterns
    TIMEFRAME         — regex: "within N days/weeks/months/years"

Usage:
    python -m nlp.ner_extractor                          # all unlabeled criteria
    python -m nlp.ner_extractor --db data/processed/trials.duckdb --batch 1000
    python -m nlp.ner_extractor --reprocess              # redo all rows
    python -m nlp.ner_extractor --spot-check             # print sample output
"""

import argparse
import re
from datetime import datetime, timezone

import duckdb


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Lab values — extended from weak_labeler._OBSERVABLE_PATTERNS
_LAB_NAMES = (
    r'hba1c|egfr|gfr|creatinine|hemoglobin|haemoglobin|platelet|wbc|alt|ast'
    r'|bilirubin|albumin|ldh|inr|aptt|ptt|psa|cea|ca125|ferritin|sodium'
    r'|potassium|calcium|magnesium|phosphate|glucose|lipase|amylase|troponin'
    r'|uric\s+acid|fibrinogen|hematocrit|haematocrit|neutrophil|lymphocyte'
    r'|eosinophil|monocyte|ejection\s+fraction|lvef|qt\b|qtc\b'
    r'|oxygen\s+saturation|spo2'
)
_LAB_RE = re.compile(rf'\b({_LAB_NAMES})\b', re.IGNORECASE)

# Thresholds — comparator + number + optional unit
_THRESHOLD_RE = re.compile(
    r'[≥≤<>]=?\s*\d+\.?\d*\s*'
    r'(?:x\s*(?:uln|upper\s+limit\s+of\s+normal)|'
    r'%|mg(?:/dl|/l|/ml)?|ml(?:/min)?|mcl|mmol(?:/l)?|mmhg|ng(?:/ml)?|'
    r'g/dl|g/l|u/l|iu/l|mm\^?3|mL/min|cm|kg(?:/m2|/m²)?|m²|years?|months?|weeks?|days?)?',
    re.IGNORECASE,
)

# Named clinical scales — name + optional numeric score
_SCALE_RE = re.compile(
    r'\b(ECOG|NYHA|CTCAE|WHO|Karnofsky|KPS|Zubrod|IPSS|NRS|PHQ|ISI|MELD|CPS|Child-Pugh)'
    r'(?:\s+(?:performance\s+status|class|grade|score|status))?'
    r'(?:\s*(?:of\s+)?[0-4]|\s*[≤≥<>]=?\s*\d)?',
    re.IGNORECASE,
)

# Demographics — age ranges, sex, BMI
_DEMO_RE = re.compile(
    r'\b(?:'
    r'\d+\s*(?:years?\s*(?:of\s+age|old)?|year[- ]old)'
    r'|age\s*[≥≤<>=\-–]\s*\d+(?:\s*(?:years?|yrs?))?'
    r'|(?:male|female|women?|men\b|sex\s*[:=]\s*\w+)'
    r'|bmi\s*[≥≤<>]\s*\d+(?:\.\d+)?'
    r'|body\s+mass\s+index\s*[≥≤<>]\s*\d+'
    r')',
    re.IGNORECASE,
)

# Timeframes
_TIMEFRAME_RE = re.compile(
    r'\b(?:'
    r'within\s+(?:the\s+)?(?:last\s+)?\d+\s*(?:days?|weeks?|months?|years?)'
    r'|in\s+the\s+(?:past|last|previous)\s+\d+\s*(?:days?|weeks?|months?|years?)'
    r'|(?:past|prior|previous|last)\s+\d+\s*(?:days?|weeks?|months?|years?)'
    r'|\d+\s*(?:days?|weeks?|months?|years?)\s+(?:prior|ago|before|previously)'
    r')',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dictionary matching helpers (CONDITION / DRUG)
# ---------------------------------------------------------------------------

def _build_trial_dicts(conn) -> dict[str, dict]:
    """
    Build per-trial condition and drug term sets for fast lookup.
    Returns: {nct_id: {'conditions': set, 'drugs': set}}
    """
    rows = conn.execute("""
        SELECT nct_id, mesh_conditions, mesh_interventions,
               conditions, interventions, intervention_other_names
        FROM trials
    """).fetchall()

    trial_dicts = {}
    for nct_id, mesh_cond, mesh_drug, raw_cond, raw_drug, aliases in rows:
        terms_cond = set()
        terms_drug = set()

        for lst in (mesh_cond, raw_cond):
            if lst:
                for t in lst:
                    if t:
                        terms_cond.add(t.lower().strip())

        for lst in (mesh_drug, raw_drug, aliases):
            if lst:
                for t in lst:
                    if t:
                        terms_drug.add(t.lower().strip())

        trial_dicts[nct_id] = {'conditions': terms_cond, 'drugs': terms_drug}

    return trial_dicts


def _match_dict_terms(text: str, terms: set[str]) -> list[str]:
    """Return all terms from `terms` found as substrings in text (lowercased)."""
    text_lower = text.lower()
    return [t for t in sorted(terms) if t and t in text_lower]


# ---------------------------------------------------------------------------
# Per-field extraction
# ---------------------------------------------------------------------------

def extract_lab_values(text: str) -> list[str]:
    return list(dict.fromkeys(m.group().strip() for m in _LAB_RE.finditer(text)))


def extract_thresholds(text: str) -> list[str]:
    return list(dict.fromkeys(m.group().strip() for m in _THRESHOLD_RE.finditer(text)))


def extract_scales(text: str) -> list[str]:
    return list(dict.fromkeys(m.group().strip() for m in _SCALE_RE.finditer(text)))


def extract_demographics(text: str) -> list[str]:
    return list(dict.fromkeys(m.group().strip() for m in _DEMO_RE.finditer(text)))


def extract_timeframes(text: str) -> list[str]:
    return list(dict.fromkeys(m.group().strip() for m in _TIMEFRAME_RE.finditer(text)))


# ---------------------------------------------------------------------------
# Main extraction function (public API)
# ---------------------------------------------------------------------------

def extract_entities(criterion_text: str, trial_dict: dict | None = None) -> dict:
    """
    Extract all entity types from a criterion text string.

    Args:
        criterion_text: raw criterion sentence
        trial_dict: optional {'conditions': set, 'drugs': set} for this trial;
                    if None, condition/drug extraction returns empty lists.

    Returns dict with keys matching the criteria table NER columns.
    """
    if trial_dict is None:
        trial_dict = {'conditions': set(), 'drugs': set()}

    return {
        'extracted_conditions':   _match_dict_terms(criterion_text, trial_dict['conditions']),
        'extracted_drugs':        _match_dict_terms(criterion_text, trial_dict['drugs']),
        'extracted_lab_values':   extract_lab_values(criterion_text),
        'extracted_thresholds':   extract_thresholds(criterion_text),
        'extracted_scales':       extract_scales(criterion_text),
        'extracted_demographics': extract_demographics(criterion_text),
        'extracted_timeframes':   extract_timeframes(criterion_text),
    }


# ---------------------------------------------------------------------------
# Batch database update
# ---------------------------------------------------------------------------

def run_extraction(db_path: str, batch_size: int, reprocess: bool) -> None:
    conn = duckdb.connect(db_path)

    print("Building trial entity dictionaries...", flush=True)
    trial_dicts = _build_trial_dicts(conn)
    print(f"  {len(trial_dicts):,} trials indexed", flush=True)

    where = (
        "WHERE text IS NOT NULL AND length(text) > 0"
        if reprocess else
        "WHERE extracted_conditions IS NULL AND text IS NOT NULL AND length(text) > 0"
    )

    total = conn.execute(f"SELECT COUNT(*) FROM criteria {where}").fetchone()[0]
    print(f"  {total:,} criteria to process", flush=True)

    # Use keyset pagination on criterion_id to avoid OFFSET shifting as rows
    # are updated in-place during incremental (non-reprocess) runs.
    processed = 0
    last_id = ''

    while True:
        rows = conn.execute(f"""
            SELECT criterion_id, nct_id, text
            FROM criteria {where}
              AND criterion_id > ?
            ORDER BY criterion_id
            LIMIT {batch_size}
        """, [last_id]).fetchall()

        if not rows:
            break

        now = datetime.now(timezone.utc)

        for criterion_id, nct_id, text in rows:
            td = trial_dicts.get(nct_id, {'conditions': set(), 'drugs': set()})
            entities = extract_entities(text, td)

            conn.execute("""
                UPDATE criteria SET
                    extracted_conditions   = ?,
                    extracted_drugs        = ?,
                    extracted_lab_values   = ?,
                    extracted_thresholds   = ?,
                    extracted_scales       = ?,
                    extracted_demographics = ?,
                    extracted_timeframes   = ?,
                    processed_at           = ?
                WHERE criterion_id = ?
            """, [
                entities['extracted_conditions'],
                entities['extracted_drugs'],
                entities['extracted_lab_values'],
                entities['extracted_thresholds'],
                entities['extracted_scales'],
                entities['extracted_demographics'],
                entities['extracted_timeframes'],
                now,
                criterion_id,
            ])

        processed += len(rows)
        last_id = rows[-1][0]
        print(f"  {processed:,}/{total:,} processed", flush=True)

    conn.close()
    print(f"\nDone. {processed:,} criteria updated.", flush=True)


# ---------------------------------------------------------------------------
# Spot-check helper
# ---------------------------------------------------------------------------

def spot_check(db_path: str, n: int = 20) -> None:
    """Print a random sample of extracted entities for manual inspection."""
    conn = duckdb.connect(db_path)
    rows = conn.execute(f"""
        SELECT text,
               extracted_conditions, extracted_drugs,
               extracted_lab_values, extracted_thresholds,
               extracted_scales, extracted_demographics,
               extracted_timeframes
        FROM criteria
        WHERE extracted_conditions IS NOT NULL
          AND (
            len(extracted_conditions) > 0
            OR len(extracted_drugs) > 0
            OR len(extracted_lab_values) > 0
          )
        USING SAMPLE {n}
    """).fetchall()
    conn.close()

    for row in rows:
        text, cond, drugs, labs, thresh, scales, demo, tf = row
        print(f"Text:         {text[:100]}")
        if cond:   print(f"  CONDITIONS: {cond}")
        if drugs:  print(f"  DRUGS:      {drugs}")
        if labs:   print(f"  LABS:       {labs}")
        if thresh: print(f"  THRESHOLDS: {thresh}")
        if scales: print(f"  SCALES:     {scales}")
        if demo:   print(f"  DEMO:       {demo}")
        if tf:     print(f"  TIMEFRAMES: {tf}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract NER entities from criteria")
    parser.add_argument('--db',         default='data/processed/trials.duckdb')
    parser.add_argument('--batch',      type=int, default=1000)
    parser.add_argument('--reprocess',  action='store_true')
    parser.add_argument('--spot-check', action='store_true',
                        help='print sample of extracted entities and exit')
    args = parser.parse_args()

    if args.spot_check:
        spot_check(args.db)
        return

    run_extraction(args.db, args.batch, args.reprocess)


if __name__ == '__main__':
    main()
