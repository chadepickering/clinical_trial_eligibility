"""
JSON → structured DuckDB records.

Extracts fields from raw ClinicalTrials.gov API v2 responses and returns
a flat dict whose keys match the trials table columns in database.py.
All list fields default to [] and all scalar fields default to None
when absent from the API response.
"""


def parse_study(raw: dict) -> dict:
    ps = raw.get("protocolSection", {})
    ds = raw.get("derivedSection", {})

    id_mod       = ps.get("identificationModule", {})
    status_mod   = ps.get("statusModule", {})
    desc_mod     = ps.get("descriptionModule", {})
    cond_mod     = ps.get("conditionsModule", {})
    design_mod   = ps.get("designModule", {})
    arms_mod     = ps.get("armsInterventionsModule", {})
    outcomes_mod = ps.get("outcomesModule", {})
    elig_mod     = ps.get("eligibilityModule", {})
    cond_browse  = ds.get("conditionBrowseModule", {})
    int_browse   = ds.get("interventionBrowseModule", {})

    interventions = arms_mod.get("interventions", [])

    return {
        # --- Identification ---
        "nct_id":       id_mod.get("nctId"),
        "brief_title":  id_mod.get("briefTitle"),

        # --- Conditions ---
        "conditions":   cond_mod.get("conditions", []),

        # --- Interventions (parallel arrays — same index = same intervention) ---
        "interventions":              [i.get("name") for i in interventions],
        "intervention_types":         [i.get("type") for i in interventions],
        "intervention_descriptions":  [i.get("description") for i in interventions],
        # otherNames is a list per intervention — flatten to one list for the trial
        "intervention_other_names":   [
            alias
            for i in interventions
            for alias in i.get("otherNames", [])
        ],

        # --- Design ---
        "phases": design_mod.get("phases", []),

        # --- Status ---
        "status": status_mod.get("overallStatus"),

        # --- Eligibility ---
        "min_age":        elig_mod.get("minimumAge"),
        "max_age":        elig_mod.get("maximumAge"),   # often absent
        "sex":            elig_mod.get("sex"),
        "std_ages":       elig_mod.get("stdAges", []),
        "eligibility_text": elig_mod.get("eligibilityCriteria"),

        # --- Outcomes ---
        "primary_outcomes": [
            o.get("measure") for o in outcomes_mod.get("primaryOutcomes", [])
        ],
        "secondary_outcomes": [
            o.get("measure") for o in outcomes_mod.get("secondaryOutcomes", [])
        ],

        # --- Descriptions ---
        "brief_summary":        desc_mod.get("briefSummary"),
        "detailed_description": desc_mod.get("detailedDescription"),  # nullable

        # --- MeSH terms (pre-normalized gold labels for NER) ---
        "mesh_conditions":    [
            m.get("term") for m in cond_browse.get("meshes", [])
        ],
        "mesh_interventions": [
            m.get("term") for m in int_browse.get("meshes", [])
        ],
    }
