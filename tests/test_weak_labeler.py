"""Tests for nlp/weak_labeler.py."""

import pytest
from nlp.weak_labeler import label_criterion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_criterion(text, section='inclusion'):
    return {'text': text, 'section': section, 'position': 0}


# ---------------------------------------------------------------------------
# B1 — Inclusion vs Exclusion
# ---------------------------------------------------------------------------

def test_b1_inclusion_section():
    result = label_criterion(make_criterion('Age >= 18 years', section='inclusion'))
    assert result['b1_label'] == 1

def test_b1_exclusion_section():
    result = label_criterion(make_criterion('Pregnant or breastfeeding', section='exclusion'))
    assert result['b1_label'] == 0

def test_b1_unknown_section_is_none():
    result = label_criterion(make_criterion('Diagnosis of osteosarcoma', section='unknown'))
    assert result['b1_label'] is None

def test_b1_does_not_modify_other_keys():
    criterion = make_criterion('Age >= 18', section='inclusion')
    result = label_criterion(criterion)
    assert result['text'] == criterion['text']
    assert result['section'] == criterion['section']
    assert result['position'] == criterion['position']


# ---------------------------------------------------------------------------
# B2 — Objective vs Subjective
# ---------------------------------------------------------------------------

def test_b2_numeric_threshold_is_objective():
    result = label_criterion(make_criterion('eGFR >= 45 mL/min/1.73m2'))
    assert result['b2_label'] == 1

def test_b2_units_of_measurement_is_objective():
    result = label_criterion(make_criterion('Hemoglobin >= 9 g/dL'))
    assert result['b2_label'] == 1

def test_b2_named_scale_with_number_is_objective():
    result = label_criterion(make_criterion('ECOG performance status 0-2'))
    assert result['b2_label'] == 1

def test_b2_uln_reference_is_objective():
    result = label_criterion(make_criterion('AST <= 2.5 x upper limit of normal'))
    assert result['b2_label'] == 1

def test_b2_normal_range_is_objective():
    result = label_criterion(make_criterion('Normal serum lipase and amylase per institutional normal values'))
    assert result['b2_label'] == 1

def test_b2_willing_is_subjective():
    result = label_criterion(make_criterion('Willing to use effective contraception during study'))
    assert result['b2_label'] == 0

def test_b2_adequate_alone_is_subjective():
    result = label_criterion(make_criterion('Adequate organ function'))
    assert result['b2_label'] == 0

def test_b2_consent_is_subjective():
    result = label_criterion(make_criterion('Patient must have signed informed consent'))
    assert result['b2_label'] == 0

def test_b2_investigator_discretion_is_subjective():
    result = label_criterion(make_criterion('Eligible at the discretion of the investigator'))
    assert result['b2_label'] == 0

def test_b2_no_signal_is_none():
    result = label_criterion(make_criterion('Prior diagnosis of rhabdomyosarcoma'))
    assert result['b2_label'] is None
    assert result['b2_confidence'] == 0.0

def test_b2_objective_wins_over_subjective():
    # "adequate" (subjective) + numeric value (objective x2) → objective wins
    result = label_criterion(make_criterion('Adequate renal function: creatinine <= 1.5 mg/dL'))
    assert result['b2_label'] == 1

def test_b2_confidence_between_zero_and_one():
    result = label_criterion(make_criterion('ECOG 0-2'))
    assert 0.0 <= result['b2_confidence'] <= 1.0


# ---------------------------------------------------------------------------
# B3 — Observable vs Unobservable
# ---------------------------------------------------------------------------

def test_b3_lab_value_is_observable():
    result = label_criterion(make_criterion('Serum creatinine <= 1.5 mg/dL'))
    assert result['b3_label'] == 1

def test_b3_ecog_is_observable():
    result = label_criterion(make_criterion('ECOG performance status of 0 to 1'))
    assert result['b3_label'] == 1

def test_b3_diagnosis_is_observable():
    result = label_criterion(make_criterion('Histologically confirmed diagnosis of glioblastoma'))
    assert result['b3_label'] == 1

def test_b3_prior_treatment_is_observable():
    result = label_criterion(make_criterion('Prior treatment with platinum-based chemotherapy'))
    assert result['b3_label'] == 1

def test_b3_metastatic_disease_is_observable():
    result = label_criterion(make_criterion('Patients with metastatic disease'))
    assert result['b3_label'] == 1

def test_b3_pregnancy_is_observable():
    result = label_criterion(make_criterion('Pregnant or breastfeeding'))
    assert result['b3_label'] == 1

def test_b3_pregnancy_test_is_observable():
    result = label_criterion(make_criterion('Negative pregnancy test for females of childbearing potential'))
    assert result['b3_label'] == 1

def test_b3_known_history_is_observable():
    result = label_criterion(make_criterion('Known history of hypersensitivity to nivolumab'))
    assert result['b3_label'] == 1

def test_b3_refractory_is_observable():
    result = label_criterion(make_criterion('Refractory to prior hypomethylating agent therapy'))
    assert result['b3_label'] == 1

def test_b3_allergy_is_observable():
    result = label_criterion(make_criterion('Known hypersensitivity or allergic reaction to study drug'))
    assert result['b3_label'] == 1

def test_b3_additional_labs_observable():
    result = label_criterion(make_criterion('Normal serum lipase and amylase'))
    assert result['b3_label'] == 1

def test_b3_cardiac_imaging_observable():
    result = label_criterion(make_criterion('Echocardiogram showing LVEF >= 50%'))
    assert result['b3_label'] == 1

def test_b3_willing_to_consent_is_unobservable():
    result = label_criterion(make_criterion('Willing to provide written informed consent'))
    assert result['b3_label'] == 0

def test_b3_life_expectancy_is_unobservable():
    result = label_criterion(make_criterion('Life expectancy of at least 12 weeks'))
    assert result['b3_label'] == 0

def test_b3_internet_access_is_unobservable():
    result = label_criterion(make_criterion('Does not have access to the internet'))
    assert result['b3_label'] == 0

def test_b3_planning_to_conceive_is_unobservable():
    result = label_criterion(make_criterion('Not planning to conceive during the study period'))
    assert result['b3_label'] == 0

def test_b3_no_signal_is_none():
    result = label_criterion(make_criterion('Having a mental illness'))
    assert result['b3_label'] is None
    assert result['b3_confidence'] == 0.0

def test_b3_confidence_between_zero_and_one():
    result = label_criterion(make_criterion('Prior chemotherapy with docetaxel'))
    assert 0.0 <= result['b3_confidence'] <= 1.0


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_output_contains_all_expected_keys():
    result = label_criterion(make_criterion('Age >= 18'))
    expected_keys = {'text', 'section', 'position', 'b1_label', 'b2_label', 'b3_label', 'b2_confidence', 'b3_confidence'}
    assert expected_keys.issubset(result.keys())

def test_original_criterion_fields_preserved():
    criterion = {'text': 'Age >= 18', 'section': 'inclusion', 'position': 5, 'extra_field': 'kept'}
    result = label_criterion(criterion)
    assert result['position'] == 5
    assert result['extra_field'] == 'kept'
