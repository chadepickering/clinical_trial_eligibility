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

    labs = patient.get("lab_values") or {}
    lab_strs = []
    lab_map = {
        "platelet_count":   ("Platelets", "/mm³", 1),
        "hemoglobin":       ("Hgb", "g/dL", 1),
        "neutrophil_count": ("ANC", "/mm³", 0),
        "creatinine":       ("Creatinine", "mg/dL", 1),
        "bilirubin":        ("Bilirubin", "mg/dL", 1),
        "alt":              ("ALT", "U/L", 0),
        "ast":              ("AST", "U/L", 0),
        "lvef":             ("LVEF", "%", 0),
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
    # Fix 2: ECOG as selectbox, KPS as free number input
    ecog_options = ["(not specified)", 0, 1, 2, 3, 4]
    ecog_raw = st.sidebar.selectbox(
        "ECOG performance status",
        options=ecog_options,
        index=2 if ex else 0,          # index 2 = value 1 for example patient
        format_func=lambda x: str(x) if x != "(not specified)" else x,
    )
    kps = st.sidebar.number_input(
        "Karnofsky (%)", min_value=0, max_value=100,
        value=80 if ex else 0, step=1,
    )

    st.sidebar.markdown("**Lab values** *(leave 0 to omit)*")
    with st.sidebar.expander("Lab values", expanded=ex):
        plat  = st.number_input("Platelet count (/mm³)", 0, 1_000_000,
                                value=180_000 if ex else 0, step=1_000)
        hgb   = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0,
                                value=12.5 if ex else 0.0, step=0.1, format="%.1f")
        anc   = st.number_input("Neutrophil count (/mm³)", 0, 50_000,
                                value=2_800 if ex else 0, step=100)
        creat = st.number_input("Creatinine (mg/dL)", 0.0, 20.0,
                                value=0.9 if ex else 0.0, step=0.1, format="%.1f")
        bili  = st.number_input("Bilirubin (mg/dL)", 0.0, 20.0,
                                value=0.7 if ex else 0.0, step=0.1, format="%.1f")
        alt   = st.number_input("ALT (U/L)", 0, 1_000,
                                value=28 if ex else 0, step=1)
        ast   = st.number_input("AST (U/L)", 0, 1_000,
                                value=22 if ex else 0, step=1)
        lvef  = st.number_input("LVEF (%)", 0, 100,
                                value=62 if ex else 0, step=1)
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

    labs: dict = {}
    if plat  > 0: labs["platelet_count"]   = float(plat)
    if hgb   > 0: labs["hemoglobin"]       = float(hgb)
    if anc   > 0: labs["neutrophil_count"] = float(anc)
    if creat > 0: labs["creatinine"]       = float(creat)
    if bili  > 0: labs["bilirubin"]        = float(bili)
    if alt   > 0: labs["alt"]              = float(alt)
    if ast   > 0: labs["ast"]              = float(ast)
    if lvef  > 0: labs["lvef"]             = float(lvef)
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
# Trial search panel
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Map similarity score 0–1 to a hex color: green (1) → yellow (0.5) → red (0)."""
    if score >= 0.5:
        # green to yellow: r increases 0→255, g stays 180
        t = (score - 0.5) / 0.5
        r = int(255 * (1 - t))
        g = int(180 * t + 200 * (1 - t))
        return f"#{r:02x}{g:02x}00"
    else:
        # yellow to red: g decreases 200→0
        t = score / 0.5
        g = int(200 * t)
        return f"#ff{g:02x}00"


def _render_search() -> str | None:
    """Renders the search panel. Returns selected nct_id or None."""
    st.subheader("Trial Search")

    # Fix 3: query box populated from full patient description
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

    st.markdown(f"**{len(results)} trials found** — select a row to score")

    import pandas as pd

    # Fix 4: correct field access (flat dict, not nested); NCT ID left, Score right
    rows = []
    for r in results:
        conds = (r.get("conditions") or "")
        # conditions is stored as a string repr of a list — clean it up
        conds = conds.strip("[]'\"").replace("', '", ", ").replace('", "', ", ")[:55]
        rows.append({
            "NCT ID":     r["nct_id"],
            "Status":     r.get("status", ""),
            "Sex":        r.get("sex", ""),
            "Min age":    r.get("min_age", ""),
            "Conditions": conds,
            "Score":      round(r["score"], 3),
        })

    df = pd.DataFrame(rows)

    # Fix 5: checkbox row selection via on_select
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score",
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
    # Count how many criteria the model could actually resolve.
    # "Evaluable" = deterministic pass/fail + subjective (has a meaningful prior).
    # "Unevaluable" = unobservable + unevaluable (Beta(3,1) placeholder priors).
    _COVERAGE_THRESHOLD = 0.4
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

    # --- Profile incomplete: not enough criteria evaluable ---
    if coverage < _COVERAGE_THRESHOLD:
        n_unobs = sum(1 for e in evaluations if e["kind"] in ("unobservable", "unevaluable"))
        st.warning(
            f"**Profile incomplete** — {n_evaluable} of {n_total} criteria evaluable "
            f"({coverage:.0%}). The eligibility probability would be unreliable with "
            f"this much missing data.\n\n"
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


def _render_criterion_table(evaluations: list[dict]):
    st.subheader("Criterion Breakdown")

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
        text = (e.get("text") or "")[:90]
        if len(e.get("text") or "") > 90:
            text += "…"
        hedging_col = (
            f'{e["hedging"]:.2f}' if kind == "subjective" else "—"
        )
        # Fix 6: explicit text color on every row so dark-mode white doesn't bleed
        # through light row backgrounds
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
            f"<td style='{td}text-align:center;'>{hedging_col}</td>"
            f"</tr>"
        )

    table_html = (
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='border-bottom:2px solid #e5e7eb;'>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Kind</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Type</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:0.82rem;'>Criterion</th>"
        "<th style='padding:4px 8px;text-align:center;font-size:0.82rem;'>Hedging</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
    )
    st.html(table_html)
    st.caption(
        "FAIL rows (red) are hard disqualifiers. "
        "SUBJ rows (amber) use hedging-shaped Beta priors. "
        "UNOBS/EVAL rows (grey) use Beta(3,1) — optimistic prior reflecting "
        "the trial-seeking population; they widen the credible interval."
    )


# ---------------------------------------------------------------------------
# LLM narrative panel
# ---------------------------------------------------------------------------

def _render_llm_panel(nct_id: str, patient: dict):
    st.subheader("AI Narrative (Mistral-7B)")

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
    _render_criterion_table(evaluations)

    st.divider()

    # 11f — LLM narrative
    _render_llm_panel(selected_nct, patient)


if __name__ == "__main__":
    main()
