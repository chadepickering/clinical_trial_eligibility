"""
Tests for bayesian/criterion_evaluator.py.

Test classes:

  TestParseThreshold          — _parse_threshold: operator/value/unit parsing
                                across common oncology criterion formats. No
                                external dependencies. Always runnable.

  TestCompare                 — _compare: all six operators, boundary values,
                                unknown operator fallback.

  TestEvaluateObjectiveCriterion
                              — evaluate_objective_criterion: each routing branch
                                (sex, ECOG, Karnofsky, age, lab values), both
                                inclusion and exclusion polarity, missing patient
                                fields, and ambiguous criteria that must return None.
                                Includes regression test for the Karnofsky / ECOG
                                "performance status" ambiguity fixed during Step 10b.

  TestEstimateHedging         — estimate_hedging: keyword coverage for high,
                                low, and default hedging tiers.

  TestParseAgeYears           — _parse_age_years: integer extraction from
                                ClinicalTrials.gov age strings ("18 Years", etc.).
                                No external dependencies. Always runnable.

  TestLoadCriteriaForTrial    — integration test: DuckDB round-trip for NCT00127920,
                                including synthetic metadata criteria (_meta_sex,
                                _meta_min_age). Skipped if the database file is absent.

  TestSyntheticCriteriaFromMetadata
                              — integration test: _synthetic_criteria_from_metadata
                                for trials with sex=FEMALE, sex=MALE, and sex=ALL.
                                Skipped if the database file is absent.

Run all unit tests (no DB):
    PYTHONPATH=. pytest tests/test_criterion_evaluator.py -v -m "not integration"

Run including integration:
    PYTHONPATH=. pytest tests/test_criterion_evaluator.py -v
"""

import os
import pytest

from bayesian.criterion_evaluator import (
    Criterion,
    _parse_threshold,
    _compare,
    _parse_age_years,
    _synthetic_criteria_from_metadata,
    evaluate_objective_criterion,
    estimate_hedging,
    load_criteria_for_trial,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "trials.duckdb"
)


def make_criterion(
    text: str,
    b1_label: int | None = 1,
    b2_label: int | None = 1,
    b3_label: int | None = 1,
    thresholds: list[str] | None = None,
    demographics: list[str] | None = None,
    lab_values: list[str] | None = None,
) -> Criterion:
    return Criterion(
        criterion_id="test_001",
        text=text,
        b1_label=b1_label,
        b2_label=b2_label,
        b3_label=b3_label,
        extracted_thresholds=thresholds or [],
        extracted_demographics=demographics or [],
        extracted_lab_values=lab_values or [],
    )


def base_patient(**overrides) -> dict:
    """Minimal qualifying patient for NCT00127920-style trials."""
    p = {
        "age": 52,
        "sex": "female",
        "ecog": 1,
        "karnofsky": 80,
        "lab_values": {
            "platelet_count": 150_000,
            "hemoglobin": 12.5,
            "neutrophil_count": 2_500,
            "creatinine": 1.0,
            "bilirubin": 0.8,
            "alt": 30,
            "ast": 25,
            "lvef": 60,
            "testosterone": 15,
        },
    }
    p.update(overrides)
    return p


# ===========================================================================
# TestParseThreshold
# ===========================================================================

class TestParseThreshold:
    """_parse_threshold: string → (operator, value, unit) or None."""

    # -- Unicode operators ---------------------------------------------------

    def test_geq_unicode_percent(self):
        assert _parse_threshold("≥ 50%") == (">=", 50.0, "%")

    def test_geq_unicode_per_mm3(self):
        assert _parse_threshold("≥ 100,000/mm³") == (">=", 100_000.0, "/mm³")

    def test_leq_unicode_mgdl(self):
        assert _parse_threshold("≤ 2.2 mg/dL") == ("<=", 2.2, "mg/dL")

    def test_geq_unicode_years(self):
        assert _parse_threshold("≥ 18 years") == (">=", 18.0, "years")

    # -- ASCII operators -----------------------------------------------------

    def test_geq_ascii(self):
        assert _parse_threshold(">= 1500/mm3") == (">=", 1500.0, "/mm3")

    def test_leq_ascii(self):
        assert _parse_threshold("<= 70 years") == ("<=", 70.0, "years")

    def test_gt_ascii(self):
        assert _parse_threshold("> 50%") == (">", 50.0, "%")

    def test_lt_ascii(self):
        assert _parse_threshold("< 90 years") == ("<", 90.0, "years")

    # -- Comma-formatted numbers ---------------------------------------------

    def test_comma_formatted_large_number(self):
        op, val, unit = _parse_threshold("≥ 100,000/mm³")
        assert val == 100_000.0

    def test_comma_formatted_platelet(self):
        op, val, unit = _parse_threshold(">= 75,000 cells/mm3")
        assert val == 75_000.0

    # -- Decimals ------------------------------------------------------------

    def test_decimal_bilirubin(self):
        assert _parse_threshold("≤ 1.5 mg/dL") == ("<=", 1.5, "mg/dL")

    def test_decimal_creatinine_ratio(self):
        op, val, unit = _parse_threshold("≤ 2.5 × ULN")
        assert op == "<="
        assert val == 2.5

    # -- Embedded in longer strings ------------------------------------------

    def test_threshold_embedded_in_sentence(self):
        result = _parse_threshold("Platelet count must be ≥ 100,000 cells/mm³ at screening")
        assert result is not None
        op, val, _ = result
        assert op == ">="
        assert val == 100_000.0

    def test_threshold_after_ecog_text(self):
        result = _parse_threshold("ECOG performance status ≤ 2")
        assert result is not None
        assert result[:2] == ("<=", 2.0)

    # -- Non-parseable inputs ------------------------------------------------

    def test_returns_none_for_empty_string(self):
        assert _parse_threshold("") is None

    def test_returns_none_for_plain_text(self):
        assert _parse_threshold("Prior chemotherapy is not permitted") is None

    def test_returns_none_for_text_only_number(self):
        # Number without operator
        assert _parse_threshold("18 years of age") is None


# ===========================================================================
# TestCompare
# ===========================================================================

class TestCompare:
    """_compare: operator application and boundary behaviour."""

    def test_geq_above_threshold(self):
        assert _compare(101.0, ">=", 100.0) is True

    def test_geq_at_threshold(self):
        assert _compare(100.0, ">=", 100.0) is True

    def test_geq_below_threshold(self):
        assert _compare(99.0, ">=", 100.0) is False

    def test_gt_strictly_above(self):
        assert _compare(51.0, ">", 50.0) is True

    def test_gt_at_threshold_is_false(self):
        assert _compare(50.0, ">", 50.0) is False

    def test_leq_at_threshold(self):
        assert _compare(2.2, "<=", 2.2) is True

    def test_leq_above_threshold(self):
        assert _compare(2.3, "<=", 2.2) is False

    def test_lt_strictly_below(self):
        assert _compare(1.9, "<", 2.0) is True

    def test_lt_at_threshold_is_false(self):
        assert _compare(2.0, "<", 2.0) is False

    def test_eq_match(self):
        assert _compare(0.0, "=", 0.0) is True

    def test_eq_no_match(self):
        assert _compare(1.0, "=", 0.0) is False

    def test_unknown_operator_returns_false(self):
        assert _compare(5.0, "??", 3.0) is False


# ===========================================================================
# TestEvaluateObjectiveCriterion
# ===========================================================================

class TestEvaluateObjectiveCriterion:
    """
    evaluate_objective_criterion: full branch coverage with both inclusion
    (b1_label=1) and exclusion (b1_label=0) polarity.

    Naming convention:
      test_<field>_<condition>_<inclusion|exclusion>
    """

    # -- Sex — inclusion -----------------------------------------------------

    def test_sex_female_required_female_patient_inclusion(self):
        c = make_criterion("Subjects must be female", b1_label=1)
        assert evaluate_objective_criterion(c, base_patient(sex="female")) is True

    def test_sex_female_required_male_patient_inclusion(self):
        c = make_criterion("Subjects must be female", b1_label=1)
        assert evaluate_objective_criterion(c, base_patient(sex="male")) is False

    def test_sex_male_required_male_patient_inclusion(self):
        c = make_criterion("Open to men only", b1_label=1)
        assert evaluate_objective_criterion(c, base_patient(sex="male")) is True

    def test_sex_male_required_female_patient_inclusion(self):
        c = make_criterion("Open to men only", b1_label=1)
        assert evaluate_objective_criterion(c, base_patient(sex="female")) is False

    # -- Sex — exclusion (polarity inversion) --------------------------------

    def test_sex_female_exclusion_female_patient(self):
        # Exclusion: "women not eligible" — female patient IS excluded → False
        c = make_criterion("Exclude: female patients", b1_label=0)
        assert evaluate_objective_criterion(c, base_patient(sex="female")) is False

    def test_sex_female_exclusion_male_patient(self):
        # Male patient does NOT trigger female exclusion → True (eligible)
        c = make_criterion("Exclude: female patients", b1_label=0)
        assert evaluate_objective_criterion(c, base_patient(sex="male")) is True

    # -- Sex — ambiguous / missing -------------------------------------------

    def test_sex_both_mentioned_returns_none(self):
        c = make_criterion("Male and female patients aged 18–75", b1_label=1)
        result = evaluate_objective_criterion(c, base_patient())
        assert result is None

    def test_sex_criterion_no_patient_sex_returns_none(self):
        c = make_criterion("Female subjects only", b1_label=1)
        patient = base_patient()
        del patient["sex"]
        assert evaluate_objective_criterion(c, patient) is None

    # -- ECOG — inclusion ----------------------------------------------------

    def test_ecog_leq2_patient_ecog1_inclusion(self):
        c = make_criterion("ECOG performance status ≤ 2", b1_label=1,
                           thresholds=["≤ 2"])
        assert evaluate_objective_criterion(c, base_patient(ecog=1)) is True

    def test_ecog_leq2_patient_ecog2_boundary_inclusion(self):
        c = make_criterion("ECOG performance status ≤ 2", b1_label=1,
                           thresholds=["≤ 2"])
        assert evaluate_objective_criterion(c, base_patient(ecog=2)) is True

    def test_ecog_leq2_patient_ecog3_inclusion(self):
        c = make_criterion("ECOG performance status ≤ 2", b1_label=1,
                           thresholds=["≤ 2"])
        assert evaluate_objective_criterion(c, base_patient(ecog=3)) is False

    def test_ecog_lt2_patient_ecog2_strict_inclusion(self):
        # ECOG < 2 means ECOG 0–1 only
        c = make_criterion("ECOG < 2 (i.e., 0–1)", b1_label=1,
                           thresholds=["< 2"])
        assert evaluate_objective_criterion(c, base_patient(ecog=2)) is False

    # -- ECOG — exclusion ----------------------------------------------------

    def test_ecog_geq3_exclusion_patient_ecog1(self):
        # Exclude ECOG ≥ 3 — patient ECOG 1 is NOT excluded → True
        c = make_criterion("Exclude patients with ECOG ≥ 3", b1_label=0,
                           thresholds=["≥ 3"])
        assert evaluate_objective_criterion(c, base_patient(ecog=1)) is True

    def test_ecog_geq3_exclusion_patient_ecog4(self):
        # Patient ECOG 4 triggers exclusion → False
        c = make_criterion("Exclude patients with ECOG ≥ 3", b1_label=0,
                           thresholds=["≥ 3"])
        assert evaluate_objective_criterion(c, base_patient(ecog=4)) is False

    # -- ECOG — missing patient field ----------------------------------------

    def test_ecog_no_patient_ecog_returns_none(self):
        c = make_criterion("ECOG ≤ 2", b1_label=1, thresholds=["≤ 2"])
        patient = base_patient()
        del patient["ecog"]
        assert evaluate_objective_criterion(c, patient) is None

    # -- Karnofsky -----------------------------------------------------------

    def test_karnofsky_gt50_patient_80_inclusion(self):
        c = make_criterion("Karnofsky performance status > 50%", b1_label=1,
                           thresholds=["> 50%"])
        assert evaluate_objective_criterion(c, base_patient(karnofsky=80)) is True

    def test_karnofsky_gt50_patient_40_inclusion(self):
        c = make_criterion("Karnofsky performance status > 50%", b1_label=1,
                           thresholds=["> 50%"])
        assert evaluate_objective_criterion(c, base_patient(karnofsky=40)) is False

    def test_karnofsky_gt50_patient_50_boundary_strict(self):
        # KPS > 50 (strict) — patient 50 fails
        c = make_criterion("Karnofsky performance status > 50%", b1_label=1,
                           thresholds=["> 50%"])
        assert evaluate_objective_criterion(c, base_patient(karnofsky=50)) is False

    def test_karnofsky_no_ecog_branch_regression(self):
        # Regression: "Karnofsky performance status > 50%" must NOT be caught
        # by the ECOG branch. With patient ecog=1 the ECOG branch would give
        # False (1 > 50 = False); the Karnofsky branch gives True (80 > 50).
        c = make_criterion("Karnofsky performance status > 50%", b1_label=1,
                           thresholds=["> 50%"])
        patient = base_patient(ecog=1, karnofsky=80)
        assert evaluate_objective_criterion(c, patient) is True

    def test_karnofsky_no_patient_karnofsky_returns_none(self):
        c = make_criterion("Karnofsky > 50%", b1_label=1, thresholds=["> 50%"])
        patient = base_patient()
        del patient["karnofsky"]
        assert evaluate_objective_criterion(c, patient) is None

    # -- Age — inclusion -----------------------------------------------------

    def test_age_geq18_patient_52_inclusion(self):
        c = make_criterion("Age ≥ 18 years", b1_label=1,
                           thresholds=["≥ 18 years"])
        assert evaluate_objective_criterion(c, base_patient(age=52)) is True

    def test_age_geq18_patient_15_inclusion(self):
        c = make_criterion("Age ≥ 18 years", b1_label=1,
                           thresholds=["≥ 18 years"])
        assert evaluate_objective_criterion(c, base_patient(age=15)) is False

    def test_age_geq65_patient_59_inclusion(self):
        c = make_criterion("Age ≥ 65 years", b1_label=1,
                           thresholds=["≥ 65 years"])
        assert evaluate_objective_criterion(c, base_patient(age=59)) is False

    def test_age_geq65_patient_65_boundary_inclusion(self):
        c = make_criterion("Age ≥ 65 years", b1_label=1,
                           thresholds=["≥ 65 years"])
        assert evaluate_objective_criterion(c, base_patient(age=65)) is True

    def test_age_leq35_min_age_exclusion_patient_31(self):
        # Exclude: age < 35 — patient 31 triggers exclusion → False
        c = make_criterion("Exclude patients below age 35 years", b1_label=0,
                           thresholds=["< 35 years"])
        assert evaluate_objective_criterion(c, base_patient(age=31)) is False

    def test_age_leq35_min_age_exclusion_patient_40(self):
        # Patient 40 does NOT trigger age < 35 exclusion → True
        c = make_criterion("Exclude patients below age 35 years", b1_label=0,
                           thresholds=["< 35 years"])
        assert evaluate_objective_criterion(c, base_patient(age=40)) is True

    def test_age_gt90_exclusion_patient_93(self):
        # Exclude age > 90 — patient 93 is excluded → False
        c = make_criterion("Patients > 90 years of age are excluded", b1_label=0,
                           thresholds=["> 90 years"])
        assert evaluate_objective_criterion(c, base_patient(age=93)) is False

    def test_age_gt90_exclusion_patient_85(self):
        c = make_criterion("Patients > 90 years of age are excluded", b1_label=0,
                           thresholds=["> 90 years"])
        assert evaluate_objective_criterion(c, base_patient(age=85)) is True

    def test_age_no_patient_age_returns_none(self):
        c = make_criterion("Age ≥ 18 years", b1_label=1, thresholds=["≥ 18 years"])
        patient = base_patient()
        del patient["age"]
        assert evaluate_objective_criterion(c, patient) is None

    # -- Lab values — platelet count -----------------------------------------

    def test_platelet_above_threshold_inclusion(self):
        c = make_criterion("Platelet count ≥ 100,000/mm³", b1_label=1,
                           thresholds=["≥ 100,000/mm³"])
        assert evaluate_objective_criterion(c, base_patient()) is True

    def test_platelet_below_threshold_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["platelet_count"] = 75_000
        c = make_criterion("Platelet count ≥ 100,000/mm³", b1_label=1,
                           thresholds=["≥ 100,000/mm³"])
        assert evaluate_objective_criterion(c, patient) is False

    def test_platelet_at_boundary_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["platelet_count"] = 100_000
        c = make_criterion("Platelet count ≥ 100,000/mm³", b1_label=1,
                           thresholds=["≥ 100,000/mm³"])
        assert evaluate_objective_criterion(c, patient) is True

    # -- Lab values — hemoglobin ---------------------------------------------

    def test_hemoglobin_above_threshold_inclusion(self):
        c = make_criterion("Hemoglobin ≥ 9.0 g/dL", b1_label=1,
                           thresholds=["≥ 9.0 g/dL"])
        assert evaluate_objective_criterion(c, base_patient()) is True

    def test_hemoglobin_below_threshold_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["hemoglobin"] = 7.5
        c = make_criterion("Hemoglobin ≥ 9.0 g/dL", b1_label=1,
                           thresholds=["≥ 9.0 g/dL"])
        assert evaluate_objective_criterion(c, patient) is False

    # -- Lab values — creatinine ---------------------------------------------

    def test_creatinine_below_threshold_inclusion(self):
        c = make_criterion("Serum creatinine ≤ 1.5 mg/dL", b1_label=1,
                           thresholds=["≤ 1.5 mg/dL"])
        assert evaluate_objective_criterion(c, base_patient()) is True

    def test_creatinine_above_threshold_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["creatinine"] = 2.0
        c = make_criterion("Serum creatinine ≤ 1.5 mg/dL", b1_label=1,
                           thresholds=["≤ 1.5 mg/dL"])
        assert evaluate_objective_criterion(c, patient) is False

    def test_creatinine_above_threshold_exclusion(self):
        # Elevated creatinine is exclusion — patient creatinine 3.4 triggers it
        patient = base_patient()
        patient["lab_values"]["creatinine"] = 3.4
        c = make_criterion("Exclude: creatinine > 2.2 mg/dL", b1_label=0,
                           thresholds=["> 2.2 mg/dL"])
        assert evaluate_objective_criterion(c, patient) is False

    def test_creatinine_normal_not_excluded(self):
        c = make_criterion("Exclude: creatinine > 2.2 mg/dL", b1_label=0,
                           thresholds=["> 2.2 mg/dL"])
        assert evaluate_objective_criterion(c, base_patient()) is True

    # -- Lab values — bilirubin ----------------------------------------------

    def test_bilirubin_above_uln_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["bilirubin"] = 2.3
        c = make_criterion("Total bilirubin ≤ 1.5 × ULN", b1_label=1,
                           thresholds=["≤ 1.5"])
        assert evaluate_objective_criterion(c, patient) is False

    # -- Lab values — LVEF ---------------------------------------------------

    def test_lvef_above_threshold_inclusion(self):
        c = make_criterion("LVEF ≥ 50%", b1_label=1, thresholds=["≥ 50%"])
        assert evaluate_objective_criterion(c, base_patient()) is True

    def test_lvef_below_threshold_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["lvef"] = 43
        c = make_criterion("LVEF ≥ 50%", b1_label=1, thresholds=["≥ 50%"])
        assert evaluate_objective_criterion(c, patient) is False

    def test_ejection_fraction_synonym_matched(self):
        # NER may store "ejection fraction" — ensure regex matches
        c = make_criterion("Left ventricular ejection fraction ≥ 50%",
                           b1_label=1, thresholds=["≥ 50%"])
        assert evaluate_objective_criterion(c, base_patient()) is True

    # -- Lab values — testosterone -------------------------------------------

    def test_testosterone_castrate_range_inclusion(self):
        # Trial requires castrate testosterone < 50 ng/dL; patient=15 → eligible
        patient = base_patient()
        patient["lab_values"]["testosterone"] = 15
        c = make_criterion("Serum testosterone < 50 ng/dL (castrate range)",
                           b1_label=1, thresholds=["< 50 ng/dL"])
        assert evaluate_objective_criterion(c, patient) is True

    def test_testosterone_not_in_castrate_range_inclusion(self):
        patient = base_patient()
        patient["lab_values"]["testosterone"] = 210
        c = make_criterion("Serum testosterone < 50 ng/dL (castrate range)",
                           b1_label=1, thresholds=["< 50 ng/dL"])
        assert evaluate_objective_criterion(c, patient) is False

    # -- Lab values — missing patient field ----------------------------------

    def test_lab_field_absent_from_patient_returns_none(self):
        patient = base_patient()
        del patient["lab_values"]["creatinine"]
        c = make_criterion("Serum creatinine ≤ 1.5 mg/dL", b1_label=1,
                           thresholds=["≤ 1.5 mg/dL"])
        assert evaluate_objective_criterion(c, patient) is None

    def test_lab_values_dict_missing_entirely_returns_none(self):
        patient = base_patient()
        del patient["lab_values"]
        c = make_criterion("Platelet count ≥ 100,000/mm³", b1_label=1,
                           thresholds=["≥ 100,000/mm³"])
        assert evaluate_objective_criterion(c, patient) is None

    # -- Threshold fallback from text ----------------------------------------

    def test_evaluates_threshold_from_text_when_list_empty(self):
        # No extracted_thresholds list — should fall back to scanning text
        c = make_criterion("ECOG ≤ 2", b1_label=1, thresholds=[])
        assert evaluate_objective_criterion(c, base_patient(ecog=1)) is True

    # -- Criteria with no matchable field → None -----------------------------

    def test_prior_chemo_criterion_returns_none(self):
        c = make_criterion(
            "No prior chemotherapy or radiotherapy is permitted",
            b1_label=1, thresholds=[],
        )
        assert evaluate_objective_criterion(c, base_patient()) is None

    def test_cancer_type_criterion_returns_none(self):
        c = make_criterion(
            "Histologically confirmed stage III/IV ovarian carcinoma",
            b1_label=1, thresholds=[],
        )
        assert evaluate_objective_criterion(c, base_patient()) is None

    def test_active_infection_exclusion_returns_none(self):
        c = make_criterion(
            "Subjects with septicemia, severe infection, or acute hepatitis",
            b1_label=0, thresholds=[],
        )
        assert evaluate_objective_criterion(c, base_patient()) is None

    def test_empty_text_returns_none(self):
        c = make_criterion("", b1_label=1, thresholds=[])
        assert evaluate_objective_criterion(c, base_patient()) is None

    # -- b1_label=None (unknown section) -------------------------------------

    def test_b1_none_not_treated_as_exclusion(self):
        # b1_label=None → is_exclusion=False → no inversion
        c = make_criterion("Female subjects only", b1_label=None)
        result = evaluate_objective_criterion(c, base_patient(sex="female"))
        assert result is True


# ===========================================================================
# TestEstimateHedging
# ===========================================================================

class TestEstimateHedging:
    """estimate_hedging: high / low / default tier classification."""

    # -- High hedging --------------------------------------------------------

    def test_willingness_is_high_hedging(self):
        assert estimate_hedging("Patient must be willing to comply") == 0.8

    def test_willingness_noun_is_high_hedging(self):
        assert estimate_hedging("Willingness to complete all study visits") == 0.8

    def test_investigator_judgment_is_high_hedging(self):
        assert estimate_hedging(
            "In the opinion of the investigator, patient is suitable"
        ) == 0.8

    def test_life_expectancy_is_high_hedging(self):
        assert estimate_hedging("Life expectancy ≥ 12 weeks") == 0.8

    def test_adequate_bone_marrow_is_high_hedging(self):
        assert estimate_hedging(
            "Subjects must have adequate bone marrow function"
        ) == 0.8

    def test_compliance_is_high_hedging(self):
        assert estimate_hedging("Patient must be able to comply with the protocol") == 0.8

    def test_physician_is_high_hedging(self):
        assert estimate_hedging(
            "Considered appropriate by the treating physician"
        ) == 0.8

    # -- Low hedging ---------------------------------------------------------

    def test_written_consent_is_low_hedging(self):
        assert estimate_hedging(
            "Signed written informed consent must be obtained"
        ) == 0.05

    def test_irb_consent_is_low_hedging(self):
        assert estimate_hedging(
            "Subjects who have signed an institutional review board (IRB) approved consent"
        ) == 0.05

    # -- Default (0.5) -------------------------------------------------------

    def test_karnofsky_criterion_is_default(self):
        # Objective criterion — no hedging keywords
        assert estimate_hedging("Karnofsky performance status > 50%") == 0.5

    def test_lab_value_criterion_is_default(self):
        assert estimate_hedging("Platelet count ≥ 100,000/mm³") == 0.5

    def test_empty_text_is_default(self):
        assert estimate_hedging("") == 0.5

    def test_ecog_criterion_is_default(self):
        assert estimate_hedging("ECOG ≤ 2") == 0.5

    # -- High hedging takes priority over low --------------------------------

    def test_high_priority_over_low_when_both_present(self):
        text = "Willing to provide signed informed consent"
        # "willing" fires high; "signed" fires low; high should win
        assert estimate_hedging(text) == 0.8


# ===========================================================================
# TestLoadCriteriaForTrial  (integration — requires DuckDB)
# ===========================================================================

@pytest.mark.integration
class TestLoadCriteriaForTrial:
    """DuckDB round-trip: load NCT00127920 and verify Criterion field types."""

    @pytest.fixture(scope="class")
    def con(self):
        duckdb = pytest.importorskip("duckdb")
        if not os.path.exists(DB_PATH):
            pytest.skip(f"Database not found: {DB_PATH}")
        conn = duckdb.connect(DB_PATH)
        yield conn
        conn.close()

    @pytest.fixture(scope="class")
    def criteria(self, con):
        return load_criteria_for_trial("NCT00127920", con)

    def test_returns_list(self, criteria):
        assert isinstance(criteria, list)

    def test_correct_count_for_nct00127920(self, criteria):
        # NCT00127920 has 10 NLP-split criteria (6 inclusion + 4 exclusion)
        # plus 2 synthetic metadata criteria (_meta_sex, _meta_min_age)
        assert len(criteria) == 12

    def test_all_elements_are_criterion_objects(self, criteria):
        for c in criteria:
            assert isinstance(c, Criterion)

    def test_criterion_ids_are_strings(self, criteria):
        for c in criteria:
            assert isinstance(c.criterion_id, str)
            assert c.criterion_id.startswith("NCT00127920_")

    def test_synthetic_criteria_present(self, criteria):
        ids = [c.criterion_id for c in criteria]
        assert "NCT00127920_meta_sex" in ids
        assert "NCT00127920_meta_min_age" in ids

    def test_synthetic_criteria_are_fully_labeled(self, criteria):
        for c in criteria:
            if "_meta_" in c.criterion_id:
                assert c.b1_label == 1
                assert c.b2_label == 1
                assert c.b3_label == 1

    def test_meta_min_age_has_threshold(self, criteria):
        min_age_c = next(
            (c for c in criteria if c.criterion_id == "NCT00127920_meta_min_age"), None
        )
        assert min_age_c is not None
        assert len(min_age_c.extracted_thresholds) > 0
        assert "18" in min_age_c.extracted_thresholds[0]

    def test_meta_sex_evaluates_correctly_for_female(self, criteria):
        sex_c = next(
            (c for c in criteria if c.criterion_id == "NCT00127920_meta_sex"), None
        )
        assert sex_c is not None
        assert evaluate_objective_criterion(sex_c, base_patient(sex="female")) is True

    def test_meta_sex_excludes_male(self, criteria):
        sex_c = next(
            (c for c in criteria if c.criterion_id == "NCT00127920_meta_sex"), None
        )
        assert sex_c is not None
        assert evaluate_objective_criterion(sex_c, base_patient(sex="male")) is False

    def test_b1_labels_are_int_or_none(self, criteria):
        for c in criteria:
            assert c.b1_label in (0, 1, None)

    def test_b1_has_both_inclusion_and_exclusion(self, criteria):
        labels = [c.b1_label for c in criteria if c.b1_label is not None]
        assert 1 in labels  # at least one inclusion
        assert 0 in labels  # at least one exclusion

    def test_b2_b3_nullable(self, criteria):
        # Some criteria have no weak label — None must be preserved, not coerced
        b2_values = {c.b2_label for c in criteria}
        assert None in b2_values

    def test_list_fields_are_lists(self, criteria):
        for c in criteria:
            assert isinstance(c.extracted_thresholds, list)
            assert isinstance(c.extracted_demographics, list)
            assert isinstance(c.extracted_lab_values, list)
            assert isinstance(c.extracted_conditions, list)
            assert isinstance(c.extracted_drugs, list)
            assert isinstance(c.extracted_scales, list)

    def test_null_list_fields_coerced_to_empty_list(self, criteria):
        # COALESCE in SQL ensures no list field is None
        for c in criteria:
            assert c.extracted_thresholds is not None

    def test_b2_confidence_is_float(self, criteria):
        for c in criteria:
            assert isinstance(c.b2_confidence, float)
            assert 0.0 <= c.b2_confidence <= 1.0

    def test_karnofsky_criterion_evaluates_correctly(self, criteria):
        # NCT00127920_4: "Karnofsky performance status > 50%" (b1=1, b2=1, b3=1)
        kps_criterion = next(
            (c for c in criteria if "Karnofsky" in c.text), None
        )
        assert kps_criterion is not None
        patient = base_patient(karnofsky=80)
        assert evaluate_objective_criterion(kps_criterion, patient) is True

    def test_karnofsky_criterion_fails_low_kps(self, criteria):
        kps_criterion = next(
            (c for c in criteria if "Karnofsky" in c.text), None
        )
        assert kps_criterion is not None
        patient = base_patient(karnofsky=40)
        assert evaluate_objective_criterion(kps_criterion, patient) is False

    def test_unknown_nct_id_returns_empty_list(self, con):
        result = load_criteria_for_trial("NCT99999999", con)
        assert result == []


# ===========================================================================
# TestParseAgeYears
# ===========================================================================

class TestParseAgeYears:
    """_parse_age_years: int extraction from ClinicalTrials.gov age strings."""

    def test_standard_format(self):
        assert _parse_age_years("18 Years") == 18

    def test_no_space(self):
        assert _parse_age_years("18Years") == 18

    def test_large_age(self):
        assert _parse_age_years("99 Years") == 99

    def test_zero_years(self):
        assert _parse_age_years("0 Years") == 0

    def test_none_input(self):
        assert _parse_age_years(None) is None

    def test_empty_string(self):
        assert _parse_age_years("") is None

    def test_n_a_string(self):
        assert _parse_age_years("N/A") is None

    def test_plain_integer_string(self):
        assert _parse_age_years("65") == 65

    def test_months_still_extracts_number(self):
        # ClinicalTrials.gov sometimes uses "6 Months" — we still extract 6
        assert _parse_age_years("6 Months") == 6

    def test_leading_text(self):
        assert _parse_age_years("Minimum: 18 Years") == 18

    def test_two_digit_age(self):
        assert _parse_age_years("25 Years") == 25

    def test_three_digit_age(self):
        assert _parse_age_years("100 Years") == 100


# ===========================================================================
# TestSyntheticCriteriaFromMetadata  (integration — requires DuckDB)
# ===========================================================================

@pytest.mark.integration
class TestSyntheticCriteriaFromMetadata:
    """
    _synthetic_criteria_from_metadata: verify Criterion objects built from
    trial-level metadata for trials with different sex and age configurations.
    """

    @pytest.fixture(scope="class")
    def con(self):
        duckdb = pytest.importorskip("duckdb")
        if not os.path.exists(DB_PATH):
            pytest.skip(f"Database not found: {DB_PATH}")
        conn = duckdb.connect(DB_PATH)
        yield conn
        conn.close()

    # -- NCT00127920: sex=FEMALE, min_age='18 Years' -------------------------

    def test_female_trial_produces_sex_criterion(self, con):
        result = _synthetic_criteria_from_metadata("NCT00127920", con)
        ids = [c.criterion_id for c in result]
        assert "NCT00127920_meta_sex" in ids

    def test_female_trial_sex_criterion_text(self, con):
        result = _synthetic_criteria_from_metadata("NCT00127920", con)
        sex_c = next(c for c in result if c.criterion_id == "NCT00127920_meta_sex")
        assert "Female" in sex_c.text

    def test_female_trial_produces_min_age_criterion(self, con):
        result = _synthetic_criteria_from_metadata("NCT00127920", con)
        ids = [c.criterion_id for c in result]
        assert "NCT00127920_meta_min_age" in ids

    def test_min_age_criterion_threshold_parseable(self, con):
        result = _synthetic_criteria_from_metadata("NCT00127920", con)
        min_age_c = next(c for c in result if c.criterion_id == "NCT00127920_meta_min_age")
        from bayesian.criterion_evaluator import _parse_threshold
        assert len(min_age_c.extracted_thresholds) > 0
        parsed = _parse_threshold(min_age_c.extracted_thresholds[0])
        assert parsed is not None
        op, val, _ = parsed
        assert op == ">="
        assert val == 18.0

    def test_all_synthetic_criteria_fully_labeled(self, con):
        result = _synthetic_criteria_from_metadata("NCT00127920", con)
        for c in result:
            assert c.b1_label == 1
            assert c.b2_label == 1
            assert c.b3_label == 1

    def test_unknown_nct_id_returns_empty(self, con):
        result = _synthetic_criteria_from_metadata("NCT99999999", con)
        assert result == []

    # -- Sex=ALL trial produces no sex criterion -----------------------------

    def test_sex_all_trial_no_sex_criterion(self, con):
        # NCT04541212 has sex=ALL — should generate no _meta_sex criterion
        result = _synthetic_criteria_from_metadata("NCT04541212", con)
        ids = [c.criterion_id for c in result]
        assert "NCT04541212_meta_sex" not in ids

    # -- Male-only trial produces MALE sex criterion -------------------------

    def test_male_trial_sex_criterion_text(self, con):
        # NCT02730975 has sex=MALE
        result = _synthetic_criteria_from_metadata("NCT02730975", con)
        sex_criteria = [c for c in result if "_meta_sex" in c.criterion_id]
        assert len(sex_criteria) == 1
        assert "Male" in sex_criteria[0].text

    def test_male_sex_criterion_excludes_female_patient(self, con):
        result = _synthetic_criteria_from_metadata("NCT02730975", con)
        sex_c = next(c for c in result if "_meta_sex" in c.criterion_id)
        assert evaluate_objective_criterion(sex_c, base_patient(sex="female")) is False

    def test_male_sex_criterion_passes_male_patient(self, con):
        result = _synthetic_criteria_from_metadata("NCT02730975", con)
        sex_c = next(c for c in result if "_meta_sex" in c.criterion_id)
        assert evaluate_objective_criterion(sex_c, base_patient(sex="male")) is True
