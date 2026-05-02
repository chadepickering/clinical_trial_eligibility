"""
Streamlit interface for the Clinical Trial Eligibility Intelligence System.

Layout:
  Sidebar  — Patient profile form (11b)
  Main col — Trial search (11c) → Bayesian score panel (11d)
           → Criterion breakdown table (11e) → LLM narrative (11f)

Run:
    streamlit run app/streamlit_app.py
"""

import hashlib
import json
import os
import re
import sys

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or app/ directory
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Cached heavy resources (process-level, survives reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_db():
    import duckdb
    return duckdb.connect(
        os.path.join(_ROOT, "data", "processed", "trials.duckdb"),
        read_only=True,
    )


@st.cache_resource
def _get_collection():
    from rag.vector_store import get_client, get_collection
    client = get_client(os.path.join(_ROOT, "data", "processed", "chroma"))
    return get_collection(client)


@st.cache_resource
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _ollama_available() -> bool:
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return any("mistral" in m for m in models)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient_hash(patient: dict) -> str:
    """Stable hash of a patient dict for cache keying."""
    return hashlib.md5(
        json.dumps(patient, sort_keys=True, default=str).encode()
    ).hexdigest()


@st.cache_data(show_spinner=False)
def _run_bayesian(nct_id: str, _patient_hash_key: str, patient_json: str):
    """Cache Bayesian scoring keyed on (nct_id, patient content).
    patient_json is the serialised patient; _patient_hash_key is the cache key."""
    from bayesian.criterion_evaluator import load_criteria_for_trial
    from bayesian.eligibility_model import (
        compute_eligibility_posterior,
        evaluate_all_criteria,
    )
    from bayesian.uncertainty import summarize_posterior, uncertainty_decomposition

    patient = json.loads(patient_json)
    con = _get_db()
    criteria = load_criteria_for_trial(nct_id, con)
    evaluations = evaluate_all_criteria(criteria, patient)
    result = compute_eligibility_posterior(criteria, patient, n_samples=2000, random_seed=42)
    summary = summarize_posterior(result)
    decomp = uncertainty_decomposition(result)
    return evaluations, result, summary, decomp


@st.cache_data(show_spinner=False)
def _search_trials(query: str, n_results: int = 10) -> list[dict]:
    model = _get_embedder()
    vec = model.encode(query, normalize_embeddings=True).tolist()
    from rag.vector_store import query_trials
    return query_trials(_get_collection(), vec, n_results=n_results)


def _build_patient_description(patient: dict) -> str:
    """Auto-generate a free-text patient description from structured fields."""
    parts = []
    sex = patient.get("sex", "")
    age = patient.get("age")
    if sex and age:
        parts.append(f"{sex.capitalize()}, {age}yo.")
    elif age:
        parts.append(f"Age {age}.")

    cancer = patient.get("cancer_type", "")
    if cancer:
        parts.append(f"{cancer}.")

    prior = patient.get("prior_therapy_notes", "")
    if prior:
        parts.append(f"{prior}.")

    ecog = patient.get("ecog")
    kps  = patient.get("karnofsky")
    if ecog is not None:
        parts.append(f"ECOG {ecog}.")
    if kps is not None:
        parts.append(f"Karnofsky {kps}%.")

    if patient.get("prior_chemo") is not None:
        parts.append("Prior chemotherapy: " + ("yes" if patient["prior_chemo"] else "none") + ".")
    if patient.get("prior_rt") is not None:
        parts.append("Prior radiation therapy: " + ("yes" if patient["prior_rt"] else "none") + ".")
    if patient.get("brain_mets") is not None:
        parts.append("Brain metastases: " + ("yes" if patient["brain_mets"] else "none") + ".")
    if patient.get("pregnant") is not None:
        parts.append("Pregnant/breastfeeding: " + ("yes" if patient["pregnant"] else "no") + ".")
    if patient.get("nyha_class") is not None:
        parts.append(f"NYHA class {patient['nyha_class']}.")
    if patient.get("child_pugh") is not None:
        parts.append(f"Child-Pugh class {patient['child_pugh']}.")

    labs = patient.get("lab_values") or {}
    lab_strs = []
    lab_map = {
        "platelet_count":   ("Platelets", "/mm³", 1),
        "hemoglobin":       ("Hgb", "g/dL", 1),
        "neutrophil_count": ("ANC", "/mm³", 0),
        "wbc":              ("WBC", "/mm³", 0),
        "inr":              ("INR", "", 2),
        "aptt":             ("aPTT", "sec", 1),
        "creatinine":       ("Creatinine", "mg/dL", 1),
        "egfr":             ("eGFR", "mL/min", 0),
        "bilirubin":        ("Bilirubin", "mg/dL", 1),
        "alt":              ("ALT", "U/L", 0),
        "ast":              ("AST", "U/L", 0),
        "albumin":          ("Albumin", "g/dL", 1),
        "lvef":             ("LVEF", "%", 0),
        "qtc":              ("QTc", "ms", 0),
        "calcium":          ("Calcium", "mg/dL", 1),
        "glucose":          ("Glucose", "mg/dL", 0),
        "potassium":        ("Potassium", "mEq/L", 1),
        "ldh":              ("LDH", "U/L", 0),
        "psa":              ("PSA", "ng/mL", 1),
        "testosterone":     ("Testosterone", "ng/dL", 0),
    }
    for key, (label, unit, decimals) in lab_map.items():
        if key in labs:
            val = labs[key]
            fmt = f"{val:,.{decimals}f}" if decimals else f"{int(val):,}"
            lab_strs.append(f"{label} {fmt} {unit}")
    if lab_strs:
        parts.append("Labs: " + ", ".join(lab_strs) + ".")

    return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Kind display helpers
# ---------------------------------------------------------------------------

_KIND_CHIP = {
    "deterministic_pass": ("✓ PASS",   "#1a7a4a", "#d4f5e2"),
    "deterministic_fail": ("✗ FAIL",   "#b91c1c", "#fee2e2"),
    "subjective":         ("~ SUBJ",   "#92400e", "#fef3c7"),
    "unobservable":       ("? UNOBS",  "#374151", "#e5e7eb"),
    "unevaluable":        ("? EVAL",   "#374151", "#e5e7eb"),
}

_TIER_COLOR = {
    "disqualified":        "#b91c1c",
    "high confidence":     "#1a7a4a",
    "moderate uncertainty":"#d97706",
    "high uncertainty":    "#92400e",
}

_TIER_BG = {
    "disqualified":        "#fee2e2",
    "high confidence":     "#d4f5e2",
    "moderate uncertainty":"#fef3c7",
    "high uncertainty":    "#ffedd5",
}


def _tier_badge(tier: str) -> str:
    color = _TIER_COLOR.get(tier, "#374151")
    bg    = _TIER_BG.get(tier, "#e5e7eb")
    return (
        f'<span style="background:{bg};color:{color};padding:3px 10px;'
        f'border-radius:12px;font-size:0.85rem;font-weight:600;">'
        f'{tier.upper()}</span>'
    )


# ---------------------------------------------------------------------------
# Patient profile sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict:
    st.sidebar.header("Patient Profile")

    if st.sidebar.button("Load example patient", use_container_width=True):
        st.session_state["_ex"] = True
    if st.sidebar.button("Clear", use_container_width=True):
        st.session_state["_ex"] = False
        st.session_state["auto_query"] = ""

    ex = st.session_state.get("_ex", False)

    # -- Clinical context ----------------------------------------------------
    cancer_type = st.sidebar.text_input(
        "Cancer type / condition",
        value="stage III ovarian carcinoma" if ex else "",
        placeholder="e.g. stage III ovarian carcinoma",
    )
    prior_therapy = st.sidebar.text_input(
        "Prior therapy notes",
        value="no prior chemotherapy or radiotherapy" if ex else "",
        placeholder="e.g. no prior chemotherapy",
    )

    st.sidebar.divider()

    # -- Structured fields ---------------------------------------------------
    sex = st.sidebar.selectbox(
        "Sex",
        options=["(not specified)", "female", "male"],
        index=1 if ex else 0,
    )
    age = st.sidebar.number_input("Age", min_value=0, max_value=120,
                                  value=52 if ex else 0, step=1)
    ecog_options = ["(not specified)", 0, 1, 2, 3, 4]
    ecog_raw = st.sidebar.selectbox(
        "ECOG performance status",
        options=ecog_options,
        index=2 if ex else 0,
        format_func=lambda x: str(x) if x != "(not specified)" else x,
    )
    kps = st.sidebar.number_input(
        "Karnofsky (%)", min_value=0, max_value=100,
        value=80 if ex else 0, step=1,
    )

    st.sidebar.markdown("**Medical history**")
    yn_opts = ["(not specified)", "Yes", "No"]
    prior_chemo_raw = st.sidebar.selectbox(
        "Prior chemotherapy", yn_opts, index=2 if ex else 0,
    )
    prior_rt_raw = st.sidebar.selectbox(
        "Prior radiation therapy", yn_opts, index=2 if ex else 0,
    )
    brain_mets_raw = st.sidebar.selectbox(
        "Brain metastases", yn_opts, index=2 if ex else 0,
    )
    pregnant_raw = st.sidebar.selectbox(
        "Pregnant / breastfeeding", yn_opts, index=2 if ex else 0,
    )
    nyha_opts = ["(not specified)", "Class I", "Class II", "Class III", "Class IV"]
    nyha_raw = st.sidebar.selectbox("NYHA class", nyha_opts, index=0)
    cp_opts = ["(not specified)", "A", "B", "C"]
    cp_raw = st.sidebar.selectbox("Child-Pugh class", cp_opts, index=0)

    with st.sidebar.expander("Lab values *(leave 0 to omit)*", expanded=ex):
        st.markdown("*Haematology*")
        plat  = st.number_input("Platelet count (/mm³)", 0, 1_000_000,
                                value=180_000 if ex else 0, step=1_000)
        hgb   = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0,
                                value=12.5 if ex else 0.0, step=0.1, format="%.1f")
        anc   = st.number_input("ANC / neutrophil count (/mm³)", 0, 50_000,
                                value=2_800 if ex else 0, step=100)
        wbc   = st.number_input("WBC (/mm³)", 0, 100_000,
                                value=0, step=100)
        st.markdown("*Coagulation*")
        inr   = st.number_input("INR", 0.0, 10.0,
                                value=0.0, step=0.1, format="%.1f")
        aptt  = st.number_input("aPTT (sec)", 0.0, 200.0,
                                value=0.0, step=1.0, format="%.1f")
        st.markdown("*Renal*")
        creat = st.number_input("Creatinine (mg/dL)", 0.0, 20.0,
                                value=0.9 if ex else 0.0, step=0.1, format="%.1f")
        egfr  = st.number_input("eGFR / CrCl (mL/min)", 0, 200,
                                value=0, step=1)
        st.markdown("*Hepatic*")
        bili  = st.number_input("Bilirubin (mg/dL)", 0.0, 20.0,
                                value=0.7 if ex else 0.0, step=0.1, format="%.1f")
        alt   = st.number_input("ALT (U/L)", 0, 1_000,
                                value=28 if ex else 0, step=1)
        ast   = st.number_input("AST (U/L)", 0, 1_000,
                                value=22 if ex else 0, step=1)
        alb   = st.number_input("Albumin (g/dL)", 0.0, 6.0,
                                value=0.0, step=0.1, format="%.1f")
        st.markdown("*Cardiac*")
        lvef  = st.number_input("LVEF (%)", 0, 100,
                                value=62 if ex else 0, step=1)
        qtc   = st.number_input("QTc (ms)", 0, 700,
                                value=0, step=1)
        st.markdown("*Metabolic / Chemistry*")
        calc  = st.number_input("Calcium (mg/dL)", 0.0, 20.0,
                                value=0.0, step=0.1, format="%.1f")
        gluc  = st.number_input("Glucose (mg/dL)", 0, 600,
                                value=0, step=1)
        pota  = st.number_input("Potassium (mEq/L)", 0.0, 10.0,
                                value=0.0, step=0.1, format="%.1f")
        ldh   = st.number_input("LDH (U/L)", 0, 5_000,
                                value=0, step=10)
        st.markdown("*Tumour markers / Reproductive*")
        psa   = st.number_input("PSA (ng/mL)", 0.0, 500.0,
                                value=0.0, step=0.1, format="%.1f")
        testo = st.number_input("Testosterone (ng/dL)", 0, 2_000,
                                value=12 if ex else 0, step=1)

    # Build patient dict — omit zero/unspecified values
    patient: dict = {}
    if cancer_type.strip():
        patient["cancer_type"] = cancer_type.strip()
    if prior_therapy.strip():
        patient["prior_therapy_notes"] = prior_therapy.strip()
    if sex != "(not specified)":
        patient["sex"] = sex
    if age > 0:
        patient["age"] = int(age)
    if ecog_raw != "(not specified)":
        patient["ecog"] = int(ecog_raw)
    if kps > 0:
        patient["karnofsky"] = int(kps)
    if prior_chemo_raw != "(not specified)":
        patient["prior_chemo"] = prior_chemo_raw == "Yes"
    if prior_rt_raw != "(not specified)":
        patient["prior_rt"] = prior_rt_raw == "Yes"
    if brain_mets_raw != "(not specified)":
        patient["brain_mets"] = brain_mets_raw == "Yes"
    if pregnant_raw != "(not specified)":
        patient["pregnant"] = pregnant_raw == "Yes"
    if nyha_raw != "(not specified)":
        patient["nyha_class"] = nyha_opts.index(nyha_raw)  # I=1, II=2, III=3, IV=4
    if cp_raw != "(not specified)":
        patient["child_pugh"] = cp_raw

    labs: dict = {}
    if plat  > 0: labs["platelet_count"]   = float(plat)
    if hgb   > 0: labs["hemoglobin"]       = float(hgb)
    if anc   > 0: labs["neutrophil_count"] = float(anc)
    if wbc   > 0: labs["wbc"]              = float(wbc)
    if inr   > 0: labs["inr"]              = float(inr)
    if aptt  > 0: labs["aptt"]             = float(aptt)
    if creat > 0: labs["creatinine"]       = float(creat)
    if egfr  > 0: labs["egfr"]             = float(egfr)
    if bili  > 0: labs["bilirubin"]        = float(bili)
    if alt   > 0: labs["alt"]              = float(alt)
    if ast   > 0: labs["ast"]              = float(ast)
    if alb   > 0: labs["albumin"]          = float(alb)
    if lvef  > 0: labs["lvef"]             = float(lvef)
    if qtc   > 0: labs["qtc"]              = float(qtc)
    if calc  > 0: labs["calcium"]          = float(calc)
    if gluc  > 0: labs["glucose"]          = float(gluc)
    if pota  > 0: labs["potassium"]        = float(pota)
    if ldh   > 0: labs["ldh"]              = float(ldh)
    if psa   > 0: labs["psa"]              = float(psa)
    if testo > 0: labs["testosterone"]     = float(testo)
    if labs:
        patient["lab_values"] = labs

    n_fields = len(patient) + len(labs)
    st.sidebar.caption(f"{n_fields} profile field{'s' if n_fields != 1 else ''} set")

    # Fix 1: button at the bottom of sidebar
    if st.sidebar.button("🔍  Find matching trials", use_container_width=True, type="primary"):
        st.session_state["auto_query"] = _build_patient_description(patient)

    return patient


# ---------------------------------------------------------------------------
# Helpers for criterion table
# ---------------------------------------------------------------------------

def _patient_context_for_criterion(ev: dict, patient: dict) -> str:
    """Return the patient value(s) most relevant to why this criterion got its kind."""
    text = (ev.get("text") or "").lower()
    kind = ev["kind"]
    labs = patient.get("lab_values") or {}
    parts = []

    if re.search(r"\bage\b|\byears?\s+(old|of\s+age)\b", text):
        age = patient.get("age")
        if age:
            parts.append(f"Age {age}")

    if re.search(r"\becog\b|\bperformance\s+status\b", text):
        ecog = patient.get("ecog")
        if ecog is not None:
            parts.append(f"ECOG {ecog}")

    if re.search(r"\bkarnofsky\b|\bkps\b", text):
        kps = patient.get("karnofsky")
        if kps is not None:
            parts.append(f"KPS {kps}%")

    if re.search(r"\bplatelet\b", text):
        plat = labs.get("platelet_count")
        if plat:
            parts.append(f"Plt {int(plat):,}/mm³")

    if re.search(r"\bhemoglobin\b|\bhgb\b|\bhaemoglobin\b", text):
        hgb = labs.get("hemoglobin")
        if hgb:
            parts.append(f"Hgb {hgb} g/dL")

    if re.search(r"\bneutrophil\b|\banc\b|\bgranulocyte\b", text):
        anc = labs.get("neutrophil_count")
        if anc:
            parts.append(f"ANC {int(anc):,}/mm³")

    if re.search(r"\bwbc\b|\bleukocyte\b|\bwhite\s+blood\s+cell\b", text):
        wbc = labs.get("wbc")
        if wbc:
            parts.append(f"WBC {int(wbc):,}/mm³")

    if re.search(r"\bcreatinine\b", text):
        creat = labs.get("creatinine")
        if creat:
            parts.append(f"Creatinine {creat} mg/dL")

    if re.search(r"\bbilirubin\b", text):
        bili = labs.get("bilirubin")
        if bili:
            parts.append(f"Bili {bili} mg/dL")

    if re.search(r"\b(alt|sgpt|alanine)\b", text):
        alt = labs.get("alt")
        if alt:
            parts.append(f"ALT {alt} U/L")

    if re.search(r"\b(ast|sgot|aspartate)\b", text):
        ast = labs.get("ast")
        if ast:
            parts.append(f"AST {ast} U/L")

    if re.search(r"\blvef\b|\bejection\s+fraction\b", text):
        lvef = labs.get("lvef")
        if lvef:
            parts.append(f"LVEF {lvef}%")

    if re.search(r"\bpregnant\b|\bpregnancy\b|\blactat\b|\bbreastfeed\b|\bnursing\b|\bchild.?bear\b", text):
        pregnant = patient.get("pregnant")
        if pregnant is None:
            parts.append("Pregnancy: not specified")
        else:
            parts.append(f"Pregnant: {'yes' if pregnant else 'no'}")

    if re.search(r"\bchemotherapy\b|\bchemo\b", text):
        pc = patient.get("prior_chemo")
        if pc is not None:
            parts.append(f"Prior chemo: {'yes' if pc else 'no'}")

    if re.search(r"\bradiation\b|\bradiotherapy\b|\birradiation\b", text):
        prt = patient.get("prior_rt")
        if prt is not None:
            parts.append(f"Prior RT: {'yes' if prt else 'no'}")

    if re.search(r"\bbrain\s+(metastas|lesion|tumor|tumour)\b|\bcns\s+metastas\b|\bintracranial\b", text):
        bm = patient.get("brain_mets")
        if bm is not None:
            parts.append(f"Brain mets: {'yes' if bm else 'no'}")

    if re.search(r"\bfemale\b|\bwomen\b|\bwoman\b|\bmale\b|\bmen\b|\bman\b|\bgender\b", text):
        sex = patient.get("sex")
        if sex:
            parts.append(f"Sex: {sex}")

    if parts:
        return "; ".join(parts)

    if kind == "unobservable":
        return "Patient data absent"
    if kind == "unevaluable":
        return "Criterion not parseable"
    return "—"


@st.cache_data(show_spinner=False)
def _fetch_trial_titles(nct_ids_json: str) -> dict[str, str]:
    """Batch-fetch brief_title from DuckDB keyed by nct_id."""
    nct_ids = json.loads(nct_ids_json)
    if not nct_ids:
        return {}
    con = _get_db()
    placeholders = ", ".join("?" for _ in nct_ids)
    rows = con.execute(
        f"SELECT nct_id, brief_title FROM trials WHERE nct_id IN ({placeholders})",
        nct_ids,
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Trial search panel
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Map similarity score 0–1 to a hex color: green (1) → yellow (0.5) → red (0)."""
    if score >= 0.5:
        t = (score - 0.5) / 0.5
        r = int(255 * (1 - t))
        g = int(180 * t + 200 * (1 - t))
        return f"#{r:02x}{g:02x}00"
    else:
        t = score / 0.5
        g = int(200 * t)
        return f"#ff{g:02x}00"


def _score_text_color(score: float) -> str:
    """Return black or white text for readability on the score cell background."""
    if score >= 0.5:
        t = (score - 0.5) / 0.5
        r_n = (255 * (1 - t)) / 255
        g_n = (180 * t + 200 * (1 - t)) / 255
    else:
        t = score / 0.5
        r_n = 1.0
        g_n = (200 * t) / 255
    # Perceived luminance (sRGB approximation)
    lum = 0.2126 * r_n + 0.7152 * g_n
    return "#111111" if lum > 0.35 else "#ffffff"


def _render_search() -> str | None:
    """Renders the search panel. Returns selected nct_id or None."""
    st.subheader("Trial Search")

    auto_query = st.session_state.get("auto_query", "")
    query = st.text_input(
        "Describe the patient or trial of interest",
        value=auto_query,
        placeholder="e.g. stage III ovarian carcinoma, no prior chemotherapy, female age 52",
        help='Click "Find matching trials" in the sidebar to auto-fill from the patient profile.',
    )

    selected_nct = st.session_state.get("selected_nct_id")

    if not query:
        st.caption(
            "Fill the patient profile in the sidebar, then click **Find matching trials** "
            "— or type a query here directly."
        )
        return selected_nct

    with st.spinner("Searching…"):
        try:
            results = _search_trials(query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return selected_nct

    if not results:
        st.warning("No results. Check that ChromaDB is populated (`python embed.py`).")
        return selected_nct

    st.caption(
        "Trials are ranked by **semantic similarity**: the patient description is encoded "
        "into a 384-dimension sentence embedding (MiniLM-L6-v2) and compared against all "
        "trials indexed in a ChromaDB vector database — nearest-neighbour retrieval, not "
        "keyword matching. Select a row to run the eligibility assessment."
    )

    import pandas as pd

    nct_ids = [r["nct_id"] for r in results]
    title_map = _fetch_trial_titles(json.dumps(nct_ids))

    rows = []
    for r in results:
        raw_title = title_map.get(r["nct_id"], "")
        short_title = (raw_title[:52] + "…") if len(raw_title) > 52 else raw_title
        rows.append({
            "NCT ID":           r["nct_id"],
            "Trial Name":       short_title,
            "Status":           r.get("status", ""),
            "Sex":              r.get("sex", ""),
            "Min age":          r.get("min_age", ""),
            "Similarity Score": round(r["score"], 3),
        })

    df = pd.DataFrame(rows)

    def _style_score(val):
        bg = _score_color(val)
        fg = _score_text_color(val)
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"

    styled = df.style.map(_style_score, subset=["Similarity Score"])

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Similarity Score": st.column_config.NumberColumn(
                "Similarity Score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )

    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        chosen = results[selected_rows[0]]["nct_id"]
        st.session_state["selected_nct_id"] = chosen
        return chosen

    return selected_nct


# ---------------------------------------------------------------------------
# Bayesian score panel
# ---------------------------------------------------------------------------

def _render_bayesian_panel(nct_id: str, patient: dict) -> tuple[list, dict] | None:
    """Returns (evaluations, result) on success, None on error."""
    import plotly.graph_objects as go

    st.subheader("Eligibility Assessment")
    st.caption(
        "Each eligibility criterion for this trial is evaluated against the patient profile "
        "and classified as a deterministic pass/fail (objectively verifiable from structured "
        "data), subjective (requires physician judgment), or unobservable (data absent from "
        "profile). A Beta-distributed prior is assigned to each criterion and combined "
        "multiplicatively via a PyMC Bayesian model to yield a posterior probability of "
        "eligibility with a 95% highest-density interval."
    )

    if not patient:
        st.info("Fill in the patient profile to compute eligibility probability.")
        return None

    ph = _patient_hash(patient)
    patient_json = json.dumps(patient, sort_keys=True)

    with st.spinner("Running Bayesian model…"):
        try:
            evaluations, result, summary, decomp = _run_bayesian(
                nct_id, ph, patient_json
            )
        except Exception as e:
            st.error(f"Bayesian scoring failed: {e}")
            return None

    # --- Coverage gate ---
    # Two conditions must both pass before showing a probability:
    #   1. Coverage ≥ 40%: enough of the criteria are evaluable (not just UNOBS)
    #   2. At least 5 total criteria: guards against trials with only 2-3 trivial
    #      metadata criteria yielding a spuriously confident P=1.0
    _COVERAGE_THRESHOLD = 0.4
    _MIN_CRITERIA = 5
    n_total = len(evaluations)
    n_evaluable = sum(
        1 for e in evaluations
        if e["kind"] in ("deterministic_pass", "deterministic_fail", "subjective")
    )
    coverage = n_evaluable / n_total if n_total > 0 else 0.0

    # --- Short-circuit: hard disqualifier ---
    if result["short_circuited"]:
        failing_id = result.get("failing_criterion", "")
        failing_text = next(
            (e["text"][:80] for e in evaluations
             if e["criterion_id"] == failing_id), ""
        )
        st.error(
            f"**⛔ Ineligible — hard disqualifier**\n\n"
            f"`{failing_id}`: {failing_text}"
        )
        _render_count_row(result)
        return evaluations, result

    # --- Profile incomplete: coverage or count too low ---
    if coverage < _COVERAGE_THRESHOLD or n_total < _MIN_CRITERIA:
        n_unobs = sum(1 for e in evaluations if e["kind"] in ("unobservable", "unevaluable"))
        reason = (
            f"only {n_total} criteria in database (minimum 5 required)"
            if n_total < _MIN_CRITERIA
            else f"{n_evaluable} of {n_total} criteria evaluable ({coverage:.0%}, threshold 40%)"
        )
        st.warning(
            f"**Profile incomplete** — {reason}. The eligibility probability would be "
            f"unreliable with this much missing data.\n\n"
            f"**{n_unobs} criteria cannot be assessed** from the current patient profile. "
            f"To improve coverage, consider adding:\n"
            f"- Cancer stage, histology, and prior therapy details (in *Cancer type* / *Prior therapy notes*)\n"
            f"- Lab values: ANC, platelets, creatinine, bilirubin, LFTs\n"
            f"- Performance status (ECOG or Karnofsky)\n\n"
            f"*The criterion breakdown below shows exactly which criteria are missing data.*"
        )
        _render_count_row(result)
        return evaluations, result

    # --- Gauge ---
    p_mean  = summary["mean"]
    p_lo    = summary["hdi_lower"]
    p_hi    = summary["hdi_upper"]
    tier    = summary["tier"]

    col_gauge, col_meta = st.columns([1, 1])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(p_mean * 100, 1),
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": _TIER_COLOR.get(tier, "#374151"), "thickness": 0.25},
                "steps": [
                    {"range": [p_lo * 100, p_hi * 100],
                     "color": _TIER_BG.get(tier, "#e5e7eb")},
                ],
                "threshold": {
                    "line": {"color": _TIER_COLOR.get(tier, "#374151"), "width": 3},
                    "thickness": 0.75,
                    "value": p_mean * 100,
                },
                "bgcolor": "white",
                "borderwidth": 0,
            },
            title={"text": "P(eligible)", "font": {"size": 14}},
        ))
        fig.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"95% HDI: {p_lo:.3f} – {p_hi:.3f} (width {summary['hdi_width']:.3f})")

    with col_meta:
        st.markdown("**Uncertainty tier**")
        st.markdown(_tier_badge(tier), unsafe_allow_html=True)
        st.markdown("")

        # Dominant source callout
        dom = decomp["dominant_source"]
        n_dom = decomp["dominant_count"]
        if dom == "unobservable":
            st.info(
                f"**{n_dom} criteria cannot be evaluated** from this patient profile. "
                "Adding lab values and cancer type would reduce uncertainty."
            )
        elif dom == "subjective":
            st.warning(
                f"**{n_dom} criteria require physician judgment** — this uncertainty "
                "cannot be resolved from structured data alone."
            )
        elif dom == "deterministic":
            st.success(
                "All evaluable criteria resolved deterministically. "
                "Remaining uncertainty comes from unresolvable criteria only."
            )
        elif dom == "unevaluable":
            st.info(
                f"**{n_dom} objective criteria could not be evaluated** — patient "
                "profile is missing the relevant fields."
            )

        _render_count_row(result)
        st.markdown(f"*{summary['explanation']}*")

    return evaluations, result


def _render_count_row(result: dict):
    cols = st.columns(4)
    cols[0].metric("✓ Deterministic", result.get("n_deterministic", 0))
    cols[1].metric("~ Subjective",    result.get("n_subjective", 0))
    cols[2].metric("? Unobservable",  result.get("n_unobservable", 0))
    cols[3].metric("? Unevaluable",   result.get("n_unevaluable", 0))


# ---------------------------------------------------------------------------
# Criterion breakdown table
# ---------------------------------------------------------------------------

_KIND_ORDER = {
    "deterministic_fail": 0,
    "deterministic_pass": 1,
    "subjective":         2,
    "unevaluable":        3,
    "unobservable":       4,
}

_B1_LABEL = {1: "INC", 0: "EXC", None: "?"}


def _render_criterion_table(evaluations: list[dict], patient: dict | None = None):
    st.subheader("Criterion Breakdown")
    st.caption(
        "Every eligibility criterion stored for this trial is listed below, sorted by "
        "evaluation outcome. The **Patient Context** column shows which patient value(s) "
        "informed the classification — or why data was absent."
    )
    st.markdown(
        "<small>"
        "**✗ FAIL** — criterion objectively not met; triggers hard disqualification &nbsp;|&nbsp; "
        "**✓ PASS** — criterion objectively met &nbsp;|&nbsp; "
        "**~ SUBJ** — subjective language (e.g. *adequate function*); assessed via hedging prior &nbsp;|&nbsp; "
        "**? UNOBS** — patient profile lacks required data &nbsp;|&nbsp; "
        "**? EVAL** — criterion text could not be parsed into a structured rule"
        "</small>",
        unsafe_allow_html=True,
    )

    if not evaluations:
        st.caption("No criteria loaded.")
        return

    sorted_evals = sorted(
        evaluations,
        key=lambda e: (_KIND_ORDER.get(e["kind"], 9), e["criterion_id"]),
    )

    rows_html = []
    for e in sorted_evals:
        kind = e["kind"]
        chip_text, chip_fg, chip_bg = _KIND_CHIP.get(kind, ("?", "#374151", "#e5e7eb"))
        chip = (
            f'<span style="background:{chip_bg};color:{chip_fg};'
            f'padding:2px 7px;border-radius:9px;font-size:0.78rem;'
            f'font-weight:600;white-space:nowrap;">{chip_text}</span>'
        )
        b1   = _B1_LABEL.get(e.get("b1_label"), "?")
        text = (e.get("text") or "")[:75]
        if len(e.get("text") or "") > 75:
            text += "…"
        hedging_col = (
            f'{e["hedging"]:.2f}' if kind == "subjective" else "—"
        )
        ctx = _patient_context_for_criterion(e, patient or {})

        if kind == "deterministic_fail":
            row_style = "background:#fff5f5;"
            text_color = "color:#7f1d1d;"
        elif kind == "deterministic_pass":
            row_style = "background:#f0faf4;"
            text_color = "color:#14532d;"
        elif kind == "subjective":
            row_style = "background:#fffbeb;"
            text_color = "color:#78350f;"
        else:
            row_style = "background:#f9fafb;"
            text_color = "color:#1f2937;"

        td = f"padding:4px 8px;font-size:0.82rem;{text_color}"
        rows_html.append(
            f'<tr style="{row_style}">'
            f"<td style='padding:4px 8px;'>{chip}</td>"
            f"<td style='{td}font-size:0.8rem;'>{b1}</td>"
            f"<td style='{td}'>{text}</td>"
            f"<td style='{td}font-size:0.78rem;color:#4b5563;'>{ctx}</td>"
            f"<td style='{td}text-align:center;'>{hedging_col}</td>"
            f"</tr>"
        )

    table_html = (
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='border-bottom:2px solid #e5e7eb;'>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Kind</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Type</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Criterion</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Patient Context</th>"
        "<th style='padding:4px 8px;text-align:center;font-size:0.82rem;'>Hedging</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
    )
    st.html(table_html)
    st.caption(
        "Type: INC = inclusion criterion, EXC = exclusion criterion. "
        "FAIL rows are hard disqualifiers regardless of other criteria. "
        "UNOBS/EVAL rows use Beta(3,1) — an optimistic prior reflecting the "
        "trial-seeking referral population — and widen the credible interval."
    )


# ---------------------------------------------------------------------------
# LLM narrative panel
# ---------------------------------------------------------------------------

def _render_llm_panel(nct_id: str, patient: dict):
    st.subheader("AI Narrative (Mistral-7B)")
    st.caption(
        "The Bayesian model excels at objective, data-driven criteria but cannot reason "
        "over free-text nuance (e.g. platinum-sensitivity windows, combination drug rules, "
        "disease-specific staging). Mistral-7B-Instruct is a 7-billion-parameter "
        "open-source LLM run **locally via Ollama** — no data leaves the machine. It reads "
        "the full trial eligibility text and produces a plain-language verdict with "
        "criterion-by-criterion reasoning. Use it as a second opinion to cross-check the "
        "Bayesian result, particularly when the Bayesian tier is 'uncertain'."
    )

    ollama_ok = _ollama_available()
    if not ollama_ok:
        st.caption(
            "Ollama is not running or mistral is not pulled. "
            "Start with `ollama serve` + `ollama pull mistral`."
        )
        return

    # Auto-generate patient description, editable
    auto_desc = _build_patient_description(patient)
    patient_desc = st.text_area(
        "Patient description (editable — add clinical context here)",
        value=auto_desc,
        height=80,
        key=f"llm_desc_{nct_id}",
    )

    if not st.button("Run AI narrative", key=f"llm_btn_{nct_id}"):
        st.caption("Click to run Mistral-7B assessment. Takes 10–30 seconds.")
        return

    col = _get_collection()
    try:
        doc_result = col.get(ids=[nct_id], include=["documents"])
        trial_doc  = doc_result["documents"][0]
    except Exception as e:
        st.error(f"Could not fetch trial document from ChromaDB: {e}")
        return

    from rag.generator import assess_trial

    with st.spinner("Generating narrative…"):
        try:
            llm_result = assess_trial(
                nct_id, trial_doc, patient_desc, temperature=0.0
            )
        except Exception as e:
            st.error(f"Ollama generation failed: {e}")
            return

    verdict = llm_result.get("verdict", "UNCERTAIN")
    verdict_color = {
        "ELIGIBLE":     "#1a7a4a",
        "NOT ELIGIBLE": "#b91c1c",
        "UNCERTAIN":    "#d97706",
    }.get(verdict, "#374151")
    verdict_bg = {
        "ELIGIBLE":     "#d4f5e2",
        "NOT ELIGIBLE": "#fee2e2",
        "UNCERTAIN":    "#fef3c7",
    }.get(verdict, "#e5e7eb")

    badge = (
        f'<span style="background:{verdict_bg};color:{verdict_color};'
        f'padding:4px 14px;border-radius:12px;font-size:0.95rem;font-weight:700;">'
        f'{verdict}</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        f"> {llm_result.get('explanation', '').strip()}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Clinical Trial Eligibility",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Clinical Trial Eligibility Intelligence System")
    st.caption("Oncology trial matching with Bayesian uncertainty quantification")

    # 11b — Patient profile sidebar
    patient = _render_sidebar()

    # 11c — Trial search
    selected_nct = _render_search()

    if not selected_nct:
        return

    st.divider()
    st.markdown(f"**Selected trial:** `{selected_nct}`")

    # 11d — Bayesian panel
    panel_result = _render_bayesian_panel(selected_nct, patient)

    if panel_result is None:
        return

    evaluations, result = panel_result

    st.divider()

    # 11e — Criterion breakdown table
    _render_criterion_table(evaluations, patient)

    st.divider()

    # 11f — LLM narrative
    _render_llm_panel(selected_nct, patient)


if __name__ == "__main__":
    main()
