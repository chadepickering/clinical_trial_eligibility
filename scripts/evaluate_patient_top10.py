"""
Full evaluation of example patient against top-10 matching trials.
Compares Bayesian model output to Mistral LLM output.

Usage:
    python scripts/evaluate_patient_top10.py
"""

import sys
import os

PROJECT_ROOT = "/Users/cepickering/Documents/Large Folders/Git/clinical_trial_eligibility"
sys.path.insert(0, PROJECT_ROOT)

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.vector_store import get_client, get_collection, query_trials
from bayesian.criterion_evaluator import load_criteria_for_trial
from bayesian.eligibility_model import compute_eligibility_posterior, evaluate_all_criteria
from bayesian.uncertainty import summarize_posterior
from rag.generator import assess_trial

# ---------------------------------------------------------------------------
# Patient definition
# ---------------------------------------------------------------------------

patient = {
    'age': 52, 'sex': 'female', 'ecog': 1, 'karnofsky': 80,
    'cancer_type': 'stage III ovarian carcinoma',
    'prior_chemo': False, 'prior_rt': False, 'brain_mets': False,
    'lab_values': {
        'platelet_count': 180000, 'hemoglobin': 12.5,
        'neutrophil_count': 2800, 'creatinine': 0.9,
        'bilirubin': 0.7, 'alt': 28, 'ast': 22, 'lvef': 62,
    }
}

PATIENT_DESCRIPTION = (
    "Female, 52yo. stage III ovarian carcinoma. no prior chemotherapy or radiotherapy. "
    "ECOG 1. Karnofsky 80%. Prior chemotherapy: none. Prior radiation therapy: none. "
    "Brain metastases: none. Labs: Platelets 180,000 /mm\u00b3, Hgb 12.5 g/dL, "
    "ANC 2,800 /mm\u00b3, Creatinine 0.9 mg/dL, Bilirubin 0.7 mg/dL, "
    "ALT 28 U/L, AST 22 U/L, LVEF 62 %."
)

# ---------------------------------------------------------------------------
# Step 1: Embed patient description and retrieve top-10 trials from ChromaDB
# ---------------------------------------------------------------------------

print("=" * 80)
print("STEP 1: Embedding patient description and querying ChromaDB")
print("=" * 80)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query_embedding = model.encode(PATIENT_DESCRIPTION, normalize_embeddings=True).tolist()

chroma_client = get_client(os.path.join(PROJECT_ROOT, "data/processed/chroma"))
collection = get_collection(chroma_client)
top10 = query_trials(collection, query_embedding, n_results=10, doc_max_len=20000)

print(f"Retrieved {len(top10)} trials from ChromaDB.\n")

# ---------------------------------------------------------------------------
# Step 2+3: Open DuckDB and run Bayesian + Mistral for each trial
# ---------------------------------------------------------------------------

con = duckdb.connect(
    os.path.join(PROJECT_ROOT, "data/processed/trials.duckdb"),
    read_only=True
)

results = []

for rank, trial_hit in enumerate(top10, start=1):
    nct_id = trial_hit["nct_id"]
    sem_score = trial_hit["score"]
    trial_document = trial_hit["document"]

    # Fetch brief_title from DuckDB
    row = con.execute(
        "SELECT brief_title FROM trials WHERE nct_id = ?", [nct_id]
    ).fetchone()
    brief_title = row[0] if row else "(title not found)"

    # ---- Bayesian model ----
    criteria = load_criteria_for_trial(nct_id, con)
    evaluations = evaluate_all_criteria(criteria, patient)
    posterior = compute_eligibility_posterior(criteria, patient)
    summary = summarize_posterior(posterior)

    n_total = len(evaluations)
    n_det_pass  = sum(1 for e in evaluations if e["kind"] == "deterministic_pass")
    n_det_fail  = sum(1 for e in evaluations if e["kind"] == "deterministic_fail")
    n_subj      = sum(1 for e in evaluations if e["kind"] == "subjective")
    n_unobs     = sum(1 for e in evaluations if e["kind"] == "unobservable")
    n_uneval    = sum(1 for e in evaluations if e["kind"] == "unevaluable")

    n_evaluable = n_det_pass + n_det_fail + n_subj
    coverage_pct = (n_evaluable / n_total * 100) if n_total > 0 else 0.0

    # ---- Mistral LLM ----
    mistral_result = assess_trial(
        nct_id=nct_id,
        trial_document=trial_document,
        patient_query=PATIENT_DESCRIPTION,
        temperature=0.0,
    )

    # ---- Agreement classification ----
    verdict = mistral_result["verdict"]
    mean_p = summary["mean"]
    tier = summary["tier"]

    if tier == "disqualified" and verdict == "NOT ELIGIBLE":
        agreement = "AGREE (both ineligible)"
    elif tier == "disqualified" and verdict == "ELIGIBLE":
        agreement = "DISAGREE (Bayesian=disqualified, Mistral=eligible)"
    elif tier == "disqualified" and verdict == "UNCERTAIN":
        agreement = "PARTIAL (Bayesian=disqualified, Mistral=uncertain)"
    elif mean_p >= 0.60 and verdict == "ELIGIBLE":
        agreement = "AGREE (both eligible)"
    elif mean_p >= 0.60 and verdict == "NOT ELIGIBLE":
        agreement = "DISAGREE (Bayesian=likely eligible, Mistral=not eligible)"
    elif mean_p >= 0.60 and verdict == "UNCERTAIN":
        agreement = "PARTIAL (Bayesian=likely eligible, Mistral=uncertain)"
    elif mean_p < 0.30 and verdict == "NOT ELIGIBLE":
        agreement = "AGREE (both lean ineligible)"
    elif mean_p < 0.30 and verdict == "ELIGIBLE":
        agreement = "DISAGREE (Bayesian=low probability, Mistral=eligible)"
    else:
        agreement = "NEUTRAL / UNCERTAIN"

    results.append({
        "rank": rank,
        "nct_id": nct_id,
        "brief_title": brief_title,
        "sem_score": sem_score,
        "evaluations": evaluations,
        "posterior": posterior,
        "summary": summary,
        "n_total": n_total,
        "n_det_pass": n_det_pass,
        "n_det_fail": n_det_fail,
        "n_subj": n_subj,
        "n_unobs": n_unobs,
        "n_uneval": n_uneval,
        "n_evaluable": n_evaluable,
        "coverage_pct": coverage_pct,
        "mistral": mistral_result,
        "agreement": agreement,
    })

con.close()

# ---------------------------------------------------------------------------
# Print detailed comparison table
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 80
THIN_SEP   = "-" * 80

print("\n\n")
print(SEPARATOR)
print("DETAILED EVALUATION REPORT: TOP-10 TRIAL MATCHES")
print(f"Patient: {PATIENT_DESCRIPTION}")
print(SEPARATOR)

for r in results:
    print(f"\n{SEPARATOR}")
    print(f"TRIAL #{r['rank']} of 10")
    print(THIN_SEP)
    print(f"NCT ID       : {r['nct_id']}")
    print(f"Title        : {r['brief_title']}")
    print(f"Semantic Score: {r['sem_score']:.4f}")

    print(f"\n--- BAYESIAN MODEL ---")
    s = r["summary"]
    p = r["posterior"]
    print(f"  P(eligible) mean   : {s['mean']:.4f}")
    print(f"  95% HDI            : [{s['hdi_lower']:.4f}, {s['hdi_upper']:.4f}]")
    print(f"  HDI width          : {s['hdi_width']:.4f}")
    print(f"  Tier               : {s['tier']}")
    print(f"  Short-circuited    : {s['short_circuited']}")
    print(f"  Failing criterion  : {s['failing_criterion'] if s['failing_criterion'] else 'None'}")
    print(f"  Bayesian explanation: {s['explanation']}")
    print(f"\n  Criterion counts:")
    print(f"    n_total           = {r['n_total']}")
    print(f"    n_deterministic   = {r['n_det_pass']}  (pass)")
    print(f"    n_det_fail        = {r['n_det_fail']}  (fail / short-circuit triggers)")
    print(f"    n_subjective      = {r['n_subj']}")
    print(f"    n_unobservable    = {r['n_unobs']}")
    print(f"    n_unevaluable     = {r['n_uneval']}")
    print(f"    n_evaluable       = {r['n_evaluable']}  (det_pass + det_fail + subjective)")
    print(f"    coverage          = {r['coverage_pct']:.1f}%")

    print(f"\n  Per-criterion breakdown:")
    for i, ev in enumerate(r["evaluations"], start=1):
        b1 = f"b1={'INC' if ev['b1_label']==1 else ('EXC' if ev['b1_label']==0 else 'UNK')}"
        b2 = f"b2={'obj' if ev['b2_label']==1 else ('subj' if ev['b2_label']==0 else 'UNK')}"
        b3 = f"b3={'obs' if ev['b3_label']==1 else ('unobs' if ev['b3_label']==0 else 'UNK')}"
        hedging_str = f"  hedging={ev['hedging']:.2f}" if ev["kind"] == "subjective" else ""
        text_short = ev["text"][:120].replace("\n", " ")
        print(f"    [{i:02d}] {ev['kind']:<20}  {b1}  {b2}  {b3}{hedging_str}")
        print(f"          \"{text_short}\"")

    print(f"\n--- MISTRAL LLM ---")
    m = r["mistral"]
    print(f"  Verdict      : {m['verdict']}")
    print(f"  Explanation  :")
    for line in m["explanation"].splitlines():
        print(f"    {line}")

    print(f"\n--- AGREEMENT ---")
    print(f"  {r['agreement']}")

print(f"\n{SEPARATOR}")
print("END OF REPORT")
print(SEPARATOR)
