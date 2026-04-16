"""Tests for nlp/criterion_splitter.py."""

import pytest
from nlp.criterion_splitter import split_criteria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sections(results):
    return [c['section'] for c in results]

def texts(results):
    return [c['text'] for c in results]


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty_list():
    assert split_criteria('') == []

def test_none_returns_empty_list():
    assert split_criteria(None) == []

def test_returns_list_of_dicts_with_required_keys():
    raw = 'Inclusion Criteria:\n\n* Age >= 18 years'
    results = split_criteria(raw)
    assert len(results) == 1
    assert set(results[0].keys()) == {'text', 'section', 'position'}


# ---------------------------------------------------------------------------
# Section assignment
# ---------------------------------------------------------------------------

def test_standard_inclusion_exclusion_sections():
    raw = (
        'Inclusion Criteria:\n\n'
        '* Age >= 18 years\n'
        '* ECOG performance status 0-2\n\n'
        'Exclusion Criteria:\n\n'
        '* Pregnant or breastfeeding\n'
        '* Prior immunotherapy treatment\n'
    )
    results = split_criteria(raw)
    assert sections(results) == ['inclusion', 'inclusion', 'exclusion', 'exclusion']

def test_all_caps_headers():
    raw = (
        'INCLUSION CRITERIA:\n\n'
        '* Histologically confirmed cancer\n\n'
        'EXCLUSION CRITERIA:\n\n'
        '* Active CNS disease\n'
    )
    results = split_criteria(raw)
    assert sections(results) == ['inclusion', 'exclusion']

def test_lowercase_headers():
    raw = (
        'Inclusion criteria:\n\n'
        '* Diagnosis of PSC\n\n'
        'Exclusion criteria:\n\n'
        '* History of liver transplantation\n'
    )
    results = split_criteria(raw)
    assert sections(results) == ['inclusion', 'exclusion']

def test_header_without_colon():
    raw = (
        'Inclusion Criteria\n\n'
        '* Age >= 18 years\n\n'
        'Exclusion Criteria\n\n'
        '* Active pregnancy confirmed\n'
    )
    results = split_criteria(raw)
    assert sections(results) == ['inclusion', 'exclusion']

def test_singular_criterion_header():
    raw = 'Inclusion Criterion\n\n* Age >= 18 years\n'
    results = split_criteria(raw)
    assert len(results) == 1
    assert results[0]['section'] == 'inclusion'

def test_disease_characteristics_header_yields_unknown():
    raw = (
        'DISEASE CHARACTERISTICS:\n\n'
        '* Diagnosis of osteosarcoma\n'
        '* Biopsy proven disease\n\n'
        'PATIENT CHARACTERISTICS:\n\n'
        '* Age >= 18 years\n'
    )
    results = split_criteria(raw)
    assert all(c['section'] == 'unknown' for c in results)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Bullet format variants
# ---------------------------------------------------------------------------

def test_numbered_bullets():
    raw = (
        'Inclusion Criteria:\n\n'
        '1. Age >= 18 years\n'
        '2. ECOG performance status 0-2\n'
        '3. Histologically confirmed diagnosis\n'
    )
    results = split_criteria(raw)
    assert len(results) == 3
    assert all(c['section'] == 'inclusion' for c in results)

def test_subpoints_folded_into_parent():
    raw = (
        'Inclusion Criteria:\n\n'
        '* Primary tumor intact:\n\n'
        '   1. Size >= 2 cm on mammography\n'
        '   2. OR >= 1 cm and ER negative\n'
        '* ECOG performance status 0-2\n'
    )
    results = split_criteria(raw)
    # First criterion should contain the sub-point text folded in
    assert 'Size >= 2 cm' in results[0]['text']
    assert 'OR >= 1 cm' in results[0]['text']
    # Second criterion is separate
    assert 'ECOG' in results[1]['text']

def test_inline_header_text_handled():
    raw = 'Inclusion Criteria:A subject will be eligible if:\n\n* Age >= 18 years\n'
    results = split_criteria(raw)
    assert len(results) >= 1
    assert results[-1]['section'] == 'inclusion'


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_short_criteria_filtered():
    raw = (
        'Inclusion Criteria:\n\n'
        '* Ok\n'           # 2 chars — filtered
        '* Age >= 18 years\n'
    )
    results = split_criteria(raw)
    assert len(results) == 1
    assert 'Age >= 18' in results[0]['text']

def test_preamble_prose_filtered():
    raw = (
        'Inclusion Criteria:\n\n'
        'Individuals must meet all of the following criteria:\n\n'
        '* Age >= 18 years\n'
        '* ECOG performance status 0-1\n'
    )
    results = split_criteria(raw)
    # Preamble should not appear as a criterion
    assert not any('must meet all' in c['text'] for c in results)
    assert len(results) == 2

def test_escaped_comparators_cleaned():
    raw = (
        'Exclusion Criteria:\n\n'
        '* BMI \\>30 kg/m2\n'
        '* Age \\<18 years at enrollment\n'
    )
    results = split_criteria(raw)
    assert '>30' in results[0]['text']
    assert '<18' in results[1]['text']
    assert '\\' not in results[0]['text']
    assert '\\' not in results[1]['text']


# ---------------------------------------------------------------------------
# Position index
# ---------------------------------------------------------------------------

def test_position_is_sequential():
    raw = (
        'Inclusion Criteria:\n\n'
        '* First criterion\n'
        '* Second criterion\n\n'
        'Exclusion Criteria:\n\n'
        '* Third criterion\n'
    )
    results = split_criteria(raw)
    assert [c['position'] for c in results] == [0, 1, 2]
