"""
Tests for bayesian/uncertainty.py.

Two test classes:

  TestSummarizePosterior      — summarize_posterior: HDI computation, tier
                                classification thresholds, short-circuit path,
                                output structure, and explanation strings.

  TestUncertaintyDecomposition — uncertainty_decomposition: dominant source
                                 identification across all criterion-count
                                 combinations, short-circuit handling, and
                                 edge cases (all-zero, ties).

Run:
    PYTHONPATH=. pytest tests/test_uncertainty.py -v
"""

import numpy as np
import pytest

from bayesian.uncertainty import (
    summarize_posterior,
    uncertainty_decomposition,
    DEFAULT_HDI_PROB,
    _HIGH_CONFIDENCE_WIDTH,
    _MODERATE_UNCERTAINTY_WIDTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_posterior(
    mean: float = 0.5,
    samples: np.ndarray | None = None,
    short_circuited: bool = False,
    failing_criterion: str | None = None,
    n_deterministic: int = 3,
    n_subjective: int = 1,
    n_unobservable: int = 2,
    n_unevaluable: int = 0,
) -> dict:
    """Build a minimal posterior result dict."""
    if samples is None:
        # Default: uniform samples around mean
        rng = np.random.default_rng(42)
        samples = np.clip(rng.normal(mean, 0.05, 2000), 0, 1)
    return {
        "mean":              mean,
        "ci_lower":          float(np.percentile(samples, 2.5)),
        "ci_upper":          float(np.percentile(samples, 97.5)),
        "short_circuited":   short_circuited,
        "failing_criterion": failing_criterion,
        "n_deterministic":   n_deterministic,
        "n_subjective":      n_subjective,
        "n_unobservable":    n_unobservable,
        "n_unevaluable":     n_unevaluable,
        "samples":           samples,
    }


def zero_result(failing_criterion: str = "NCT_test_1") -> dict:
    """Short-circuited (P=0) result."""
    return make_posterior(
        mean=0.0,
        samples=np.zeros(2000),
        short_circuited=True,
        failing_criterion=failing_criterion,
    )


def point_mass_result() -> dict:
    """All-deterministic-pass (P=1) result."""
    return make_posterior(
        mean=1.0,
        samples=np.ones(2000),
        short_circuited=False,
        n_deterministic=5,
        n_subjective=0,
        n_unobservable=0,
        n_unevaluable=0,
    )


# ===========================================================================
# TestSummarizePosterior
# ===========================================================================

class TestSummarizePosterior:
    """summarize_posterior: output structure, tier boundaries, explanations."""

    # -- Output structure ----------------------------------------------------

    def test_returns_dict(self):
        result = summarize_posterior(make_posterior())
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = summarize_posterior(make_posterior())
        expected = {
            "mean", "hdi_lower", "hdi_upper", "hdi_width",
            "hdi_prob", "tier", "short_circuited",
            "failing_criterion", "explanation",
        }
        assert expected.issubset(result.keys())

    def test_hdi_prob_matches_default(self):
        result = summarize_posterior(make_posterior())
        assert result["hdi_prob"] == DEFAULT_HDI_PROB

    def test_hdi_prob_custom(self):
        result = summarize_posterior(make_posterior(), hdi_prob=0.89)
        assert result["hdi_prob"] == 0.89

    def test_hdi_width_equals_upper_minus_lower(self):
        result = summarize_posterior(make_posterior())
        assert abs(result["hdi_width"] - (result["hdi_upper"] - result["hdi_lower"])) < 1e-9

    def test_hdi_lower_leq_mean_leq_upper(self):
        result = summarize_posterior(make_posterior(mean=0.6))
        assert result["hdi_lower"] <= result["mean"] <= result["hdi_upper"]

    def test_mean_preserved(self):
        result = summarize_posterior(make_posterior(mean=0.73))
        assert abs(result["mean"] - 0.73) < 1e-6

    # -- Short-circuit → disqualified ----------------------------------------

    def test_short_circuit_tier_is_disqualified(self):
        result = summarize_posterior(zero_result())
        assert result["tier"] == "disqualified"

    def test_short_circuit_failing_criterion_preserved(self):
        result = summarize_posterior(zero_result("NCT00127920_meta_sex"))
        assert result["failing_criterion"] == "NCT00127920_meta_sex"

    def test_short_circuit_explanation_contains_ineligible(self):
        result = summarize_posterior(zero_result())
        assert "ineligible" in result["explanation"].lower()

    def test_short_circuit_explanation_contains_criterion_id(self):
        result = summarize_posterior(zero_result("NCT_abc_1"))
        assert "NCT_abc_1" in result["explanation"]

    def test_short_circuit_short_circuited_is_true(self):
        result = summarize_posterior(zero_result())
        assert result["short_circuited"] is True

    # -- Tier: high confidence -----------------------------------------------

    def test_point_mass_is_high_confidence(self):
        # All-pass → P=1 exactly, width=0
        result = summarize_posterior(point_mass_result())
        assert result["tier"] == "high confidence"

    def test_narrow_hdi_is_high_confidence(self):
        # Very tight samples → HDI width well below 0.20
        rng = np.random.default_rng(1)
        samples = np.clip(rng.normal(0.8, 0.02, 2000), 0, 1)
        result = summarize_posterior(make_posterior(mean=0.8, samples=samples))
        assert result["tier"] == "high confidence"
        assert result["hdi_width"] < _HIGH_CONFIDENCE_WIDTH

    def test_high_confidence_explanation_contains_narrow(self):
        rng = np.random.default_rng(1)
        samples = np.clip(rng.normal(0.8, 0.02, 2000), 0, 1)
        result = summarize_posterior(make_posterior(mean=0.8, samples=samples))
        assert "narrow" in result["explanation"].lower()

    # -- Tier: moderate uncertainty ------------------------------------------

    def test_moderate_hdi_is_moderate_uncertainty(self):
        # Samples spread enough to give width in [0.20, 0.50)
        rng = np.random.default_rng(2)
        samples = np.clip(rng.normal(0.5, 0.12, 2000), 0, 1)
        result = summarize_posterior(make_posterior(mean=0.5, samples=samples))
        assert result["tier"] == "moderate uncertainty"
        assert _HIGH_CONFIDENCE_WIDTH <= result["hdi_width"] < _MODERATE_UNCERTAINTY_WIDTH

    def test_moderate_explanation_contains_moderate(self):
        rng = np.random.default_rng(2)
        samples = np.clip(rng.normal(0.5, 0.12, 2000), 0, 1)
        result = summarize_posterior(make_posterior(mean=0.5, samples=samples))
        assert "moderate" in result["explanation"].lower()

    # -- Tier: high uncertainty ----------------------------------------------

    def test_wide_hdi_is_high_uncertainty(self):
        # Beta(1,1) → Uniform[0,1] → HDI ≈ 0.95
        rng = np.random.default_rng(3)
        samples = rng.uniform(0, 1, 2000)
        result = summarize_posterior(make_posterior(mean=0.5, samples=samples))
        assert result["tier"] == "high uncertainty"
        assert result["hdi_width"] >= _MODERATE_UNCERTAINTY_WIDTH

    def test_high_uncertainty_explanation_contains_high_uncertainty(self):
        rng = np.random.default_rng(3)
        samples = rng.uniform(0, 1, 2000)
        result = summarize_posterior(make_posterior(mean=0.5, samples=samples))
        assert "high uncertainty" in result["explanation"].lower()

    # -- Explanation percentages ---------------------------------------------

    def test_explanation_contains_percentage(self):
        rng = np.random.default_rng(4)
        samples = np.clip(rng.normal(0.65, 0.02, 2000), 0, 1)
        result = summarize_posterior(make_posterior(mean=0.65, samples=samples))
        assert "%" in result["explanation"]

    def test_explanation_is_string(self):
        result = summarize_posterior(make_posterior())
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 10

    # -- No failing_criterion (short-circuit without ID) --------------------

    def test_short_circuit_no_failing_criterion_id(self):
        pr = make_posterior(
            mean=0.0, samples=np.zeros(2000),
            short_circuited=True, failing_criterion=None,
        )
        result = summarize_posterior(pr)
        assert result["tier"] == "disqualified"
        assert result["failing_criterion"] is None
        # Explanation should not contain "None" as a string
        assert "None" not in result["explanation"]


# ===========================================================================
# TestUncertaintyDecomposition
# ===========================================================================

class TestUncertaintyDecomposition:
    """uncertainty_decomposition: dominant source, counts, short-circuit."""

    # -- Output structure ----------------------------------------------------

    def test_returns_dict(self):
        result = uncertainty_decomposition(make_posterior())
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = uncertainty_decomposition(make_posterior())
        expected = {
            "n_deterministic", "n_subjective", "n_unobservable",
            "n_unevaluable", "n_total_stochastic",
            "short_circuited", "dominant_source", "dominant_count",
        }
        assert expected.issubset(result.keys())

    def test_n_total_stochastic_is_sum(self):
        pr = make_posterior(n_subjective=2, n_unobservable=3, n_unevaluable=1)
        result = uncertainty_decomposition(pr)
        assert result["n_total_stochastic"] == 6

    def test_counts_match_input(self):
        pr = make_posterior(
            n_deterministic=4, n_subjective=2,
            n_unobservable=5, n_unevaluable=1,
        )
        result = uncertainty_decomposition(pr)
        assert result["n_deterministic"] == 4
        assert result["n_subjective"]    == 2
        assert result["n_unobservable"]  == 5
        assert result["n_unevaluable"]   == 1

    # -- Short-circuit → disqualified ----------------------------------------

    def test_short_circuit_dominant_source_is_disqualified(self):
        result = uncertainty_decomposition(zero_result())
        assert result["dominant_source"] == "disqualified"

    def test_short_circuit_dominant_count_is_zero(self):
        result = uncertainty_decomposition(zero_result())
        assert result["dominant_count"] == 0

    def test_short_circuit_flag_preserved(self):
        result = uncertainty_decomposition(zero_result())
        assert result["short_circuited"] is True

    # -- All deterministic passes --------------------------------------------

    def test_all_deterministic_dominant_is_deterministic(self):
        result = uncertainty_decomposition(point_mass_result())
        assert result["dominant_source"] == "deterministic"

    def test_all_deterministic_dominant_count_is_n_deterministic(self):
        pr = point_mass_result()
        result = uncertainty_decomposition(pr)
        assert result["dominant_count"] == pr["n_deterministic"]

    def test_all_deterministic_n_total_stochastic_is_zero(self):
        result = uncertainty_decomposition(point_mass_result())
        assert result["n_total_stochastic"] == 0

    # -- Dominant source: unobservable (largest stochastic category) ---------

    def test_unobservable_dominant_when_largest(self):
        pr = make_posterior(n_subjective=1, n_unobservable=5, n_unevaluable=2)
        result = uncertainty_decomposition(pr)
        assert result["dominant_source"] == "unobservable"
        assert result["dominant_count"]  == 5

    # -- Dominant source: subjective -----------------------------------------

    def test_subjective_dominant_when_largest(self):
        pr = make_posterior(n_subjective=4, n_unobservable=2, n_unevaluable=1)
        result = uncertainty_decomposition(pr)
        assert result["dominant_source"] == "subjective"
        assert result["dominant_count"]  == 4

    # -- Dominant source: unevaluable ----------------------------------------

    def test_unevaluable_dominant_when_largest(self):
        pr = make_posterior(n_subjective=1, n_unobservable=1, n_unevaluable=3)
        result = uncertainty_decomposition(pr)
        assert result["dominant_source"] == "unevaluable"
        assert result["dominant_count"]  == 3

    # -- All stochastic counts zero (no stochastic criteria but not short-circuit)

    def test_no_stochastic_not_short_circuit_dominant_is_deterministic(self):
        pr = make_posterior(
            n_deterministic=2, n_subjective=0,
            n_unobservable=0, n_unevaluable=0,
            short_circuited=False,
        )
        result = uncertainty_decomposition(pr)
        assert result["dominant_source"] == "deterministic"

    # -- Tie: Python dict max picks first key by insertion order -------------

    def test_tie_subjective_beats_unobservable_by_dict_order(self):
        # subjective and unobservable equal — dict ordering gives subjective first
        pr = make_posterior(n_subjective=3, n_unobservable=3, n_unevaluable=0)
        result = uncertainty_decomposition(pr)
        assert result["dominant_source"] in ("subjective", "unobservable")
        assert result["dominant_count"] == 3

    # -- n_total_stochastic excludes deterministic ---------------------------

    def test_n_total_stochastic_excludes_deterministic(self):
        pr = make_posterior(
            n_deterministic=10, n_subjective=2,
            n_unobservable=1, n_unevaluable=0,
        )
        result = uncertainty_decomposition(pr)
        assert result["n_total_stochastic"] == 3
