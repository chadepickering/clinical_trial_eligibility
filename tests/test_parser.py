"""Tests for ingestion/parser.py."""

from ingestion.parser import parse_study


# ---------------------------------------------------------------------------
# Fixtures — reusable raw study dicts
# ---------------------------------------------------------------------------

FULL_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "A Test Oncology Trial",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
        },
        "descriptionModule": {
            "briefSummary": "A brief summary of the trial.",
            "detailedDescription": "A longer detailed description.",
        },
        "conditionsModule": {
            "conditions": ["Glioblastoma", "Brain Tumor"],
        },
        "designModule": {
            "phases": ["PHASE2"],
        },
        "armsInterventionsModule": {
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Nivolumab",
                    "description": "IV over 30 minutes on Day 1.",
                    "otherNames": ["anti-PD1", "BMS-936558"],
                },
                {
                    "type": "DRUG",
                    "name": "Ipilimumab",
                    "description": "IV over 90 minutes on Day 1.",
                    "otherNames": ["anti-CTLA4"],
                },
            ],
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {"measure": "Overall survival", "timeFrame": "24 months"},
            ],
            "secondaryOutcomes": [
                {"measure": "Progression-free survival", "timeFrame": "12 months"},
                {"measure": "ECOG performance status", "timeFrame": "6 months"},
            ],
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Inclusion Criteria:\n* Age >= 18\nExclusion Criteria:\n* Pregnancy",
            "minimumAge": "18 Years",
            "maximumAge": "75 Years",
            "sex": "ALL",
            "stdAges": ["ADULT", "OLDER_ADULT"],
        },
    },
    "derivedSection": {
        "conditionBrowseModule": {
            "meshes": [
                {"id": "D005909", "term": "Glioblastoma"},
            ],
        },
        "interventionBrowseModule": {
            "meshes": [
                {"id": "D000077594", "term": "Nivolumab"},
                {"id": "D000074324", "term": "Ipilimumab"},
            ],
        },
    },
}

MINIMAL_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000002",
        },
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_full_study():
    """All fields extracted correctly from a fully populated study."""
    result = parse_study(FULL_STUDY)

    assert result["nct_id"] == "NCT00000001"
    assert result["brief_title"] == "A Test Oncology Trial"
    assert result["conditions"] == ["Glioblastoma", "Brain Tumor"]
    assert result["phases"] == ["PHASE2"]
    assert result["status"] == "RECRUITING"
    assert result["min_age"] == "18 Years"
    assert result["max_age"] == "75 Years"
    assert result["sex"] == "ALL"
    assert result["std_ages"] == ["ADULT", "OLDER_ADULT"]
    assert result["brief_summary"] == "A brief summary of the trial."
    assert result["detailed_description"] == "A longer detailed description."
    assert "Inclusion Criteria" in result["eligibility_text"]


def test_parse_minimal_study():
    """Optional fields default to None or [] when absent."""
    result = parse_study(MINIMAL_STUDY)

    assert result["nct_id"] == "NCT00000002"
    assert result["brief_title"] is None
    assert result["conditions"] == []
    assert result["interventions"] == []
    assert result["intervention_types"] == []
    assert result["intervention_descriptions"] == []
    assert result["intervention_other_names"] == []
    assert result["phases"] == []
    assert result["status"] is None
    assert result["min_age"] is None
    assert result["max_age"] is None
    assert result["sex"] is None
    assert result["std_ages"] == []
    assert result["primary_outcomes"] == []
    assert result["secondary_outcomes"] == []
    assert result["brief_summary"] is None
    assert result["detailed_description"] is None
    assert result["eligibility_text"] is None
    assert result["mesh_conditions"] == []
    assert result["mesh_interventions"] == []


def test_parse_intervention_parallel_arrays():
    """interventions, intervention_types, and intervention_descriptions are the same length."""
    result = parse_study(FULL_STUDY)

    assert len(result["interventions"]) == 2
    assert len(result["intervention_types"]) == 2
    assert len(result["intervention_descriptions"]) == 2
    assert result["interventions"] == ["Nivolumab", "Ipilimumab"]
    assert result["intervention_types"] == ["DRUG", "DRUG"]
    assert result["intervention_descriptions"] == [
        "IV over 30 minutes on Day 1.",
        "IV over 90 minutes on Day 1.",
    ]


def test_parse_intervention_other_names_flattened():
    """otherNames lists across multiple interventions are flattened into one list."""
    result = parse_study(FULL_STUDY)

    # Nivolumab has 2 aliases, Ipilimumab has 1 — expect 3 total, flattened
    assert result["intervention_other_names"] == ["anti-PD1", "BMS-936558", "anti-CTLA4"]


def test_parse_max_age_nullable():
    """max_age is None when maximumAge is absent from the response."""
    result = parse_study(MINIMAL_STUDY)
    assert result["max_age"] is None


def test_parse_mesh_terms():
    """MeSH condition and intervention terms extracted from derivedSection."""
    result = parse_study(FULL_STUDY)

    assert result["mesh_conditions"] == ["Glioblastoma"]
    assert result["mesh_interventions"] == ["Nivolumab", "Ipilimumab"]


def test_parse_outcomes():
    """Primary and secondary outcomes extracted as measure strings only."""
    result = parse_study(FULL_STUDY)

    assert result["primary_outcomes"] == ["Overall survival"]
    assert result["secondary_outcomes"] == [
        "Progression-free survival",
        "ECOG performance status",
    ]


def test_parse_sex_field():
    """Non-ALL sex value is correctly extracted."""
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000003"},
            "eligibilityModule": {"sex": "FEMALE"},
        }
    }
    result = parse_study(study)
    assert result["sex"] == "FEMALE"
