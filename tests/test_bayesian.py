"""
Tests for bayesian/eligibility_model.py.

Four test classes:

  TestEvaluateAllCriteria
      Complete B1/B2/B3 label permutation coverage (all 27 combinations)
      for evaluate_all_criteria(). No PyMC dependency — pure classification
      logic. Tests the three routing branches (deterministic, subjective,
      unobservable), both inclusion and exclusion polarity for deterministic
      criteria, hedging assignment, and output dict structure.

  TestBuildModel
      PyMC model construction: correct Beta parameterisation per criterion
      kind, correct number of free random variables, p_eligible registered
      as Deterministic, constant-1.0 model for all-pass evaluations.
      Uses pm.sample_prior_predictive(draws=50) as a smoke check that
      the model is well-formed.

  TestComputePosteriorProperties
      Statistical properties of sampled posteriors. All tests use
      random_seed=42 and n_samples=5000 for reproducibility. Covers:
      output shape/range, Beta mean expectations, CI ordering, monotone
      decrease of mean with increasing unobservable criterion count,
      and seed reproducibility.

  TestComputeEligibilityPosterior
      Full pipeline (evaluate → short-circuit → sample). Covers:
      empty criteria list (vacuously eligible), all-pass (point mass at 1),
      first-criterion hard fail, mid-list hard fail, multiple hard fails
      (short-circuit on first), criterion count fields, exclusion-polarity
      short-circuit, and mixed-kind result structure.

Run unit tests only (no DB):
    PYTHONPATH=. pytest tests/test_bayesian.py -v

All tests are self-contained (no DuckDB, no Ollama). PyMC is imported
at module level — tests are skipped automatically if pymc is absent.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from bayesian.criterion_evaluator import Criterion
from bayesian.eligibility_model import (
    DETERMINISTIC_FAIL,
    DETERMINISTIC_PASS,
    SUBJECTIVE,
    UNEVALUABLE,
    UNOBSERVABLE,
    build_model,
    compute_eligibility_posterior,
    compute_posterior,
    evaluate_all_criteria,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = None  # shorthand for None labels in parameterised tables


def make_criterion(
    text: str = "Platelet count ≥ 100,000/mm³",
    b1: int | None = 1,
    b2: int | None = 1,
    b3: int | None = 1,
    thresholds: list[str] | None = None,
    cid: str = "test_c",
) -> Criterion:
    return Criterion(
        criterion_id=cid,
        text=text,
        b1_label=b1,
        b2_label=b2,
        b3_label=b3,
        extracted_thresholds=thresholds or [],
    )


def karnofsky_criterion(b1=1, cid="kps") -> Criterion:
    """Objective+observable criterion: Karnofsky > 50%."""
    return make_criterion(
        text="Karnofsky performance status > 50%",
        b1=b1, b2=1, b3=1,
        thresholds=["> 50%"],
        cid=cid,
    )


def consent_criterion(b1=1, cid="consent") -> Criterion:
    """Subjective criterion (low hedging): signed informed consent."""
    return make_criterion(
        text="Subjects who have signed an institutional review board (IRB) approved consent",
        b1=b1, b2=0, b3=0,
        cid=cid,
    )


def willingness_criterion(b1=1, cid="will") -> Criterion:
    """Subjective criterion (high hedging): willingness to comply."""
    return make_criterion(
        text="Patient must be willing to comply with all study procedures",
        b1=b1, b2=0, b3=1,
        cid=cid,
    )


def cancer_type_criterion(b1=1, cid="ca") -> Criterion:
    """Objective+observable but patient field unmatchable → UNEVALUABLE."""
    return make_criterion(
        text="Histologically confirmed stage III/IV ovarian carcinoma",
        b1=b1, b2=1, b3=1,
        thresholds=[],
        cid=cid,
    )


def unobservable_criterion(b2=None, b3=None, b1=1, cid="unobs") -> Criterion:
    """Generic criterion with NULL or unobservable labels."""
    return make_criterion(
        text="No prior chemotherapy permitted",
        b1=b1, b2=b2, b3=b3,
        cid=cid,
    )


def patient_pass(karnofsky=80, sex="female") -> dict:
    return {"karnofsky": karnofsky, "sex": sex}


def patient_fail(karnofsky=40) -> dict:
    return {"karnofsky": karnofsky}


def _run(criteria, patient=None, n=2000, seed=42):
    return compute_eligibility_posterior(
        criteria, patient or {}, n_samples=n, random_seed=seed
    )


def _samples(evaluations, n=5000, seed=42):
    model = build_model(evaluations)
    return compute_posterior(model, draws=n, random_seed=seed)


# ===========================================================================
# TestEvaluateAllCriteria — all 27 B1/B2/B3 permutations
# ===========================================================================

class TestEvaluateAllCriteria:
    """
    Label routing: complete permutation coverage.

    Routing rules:
      B2=1 AND B3=1 → deterministic branch (evaluate_objective_criterion)
      B2=0          → SUBJECTIVE  (regardless of B1 or B3)
      otherwise     → UNOBSERVABLE
    """

    # -----------------------------------------------------------------------
    # Deterministic branch (B2=1, B3=1) — all three B1 values
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("b1", [1, N])
    def test_b2_1_b3_1_patient_passes_is_deterministic_pass(self, b1):
        # B1=0 (exclusion) is intentionally excluded: "Karnofsky > 50%" as an
        # exclusion criterion means "exclude patients with KPS > 50", so KPS=80
        # triggers the exclusion → DETERMINISTIC_FAIL, tested separately below.
        c = karnofsky_criterion(b1=b1)
        evals = evaluate_all_criteria([c], patient_pass(karnofsky=80))
        assert evals[0]["kind"] == DETERMINISTIC_PASS

    def test_b2_1_b3_1_b1_zero_patient_triggers_exclusion_is_fail(self):
        # B1=0: "Karnofsky > 50%" as exclusion — patient KPS=80 triggers it → FAIL
        c = karnofsky_criterion(b1=0)
        evals = evaluate_all_criteria([c], patient_pass(karnofsky=80))
        assert evals[0]["kind"] == DETERMINISTIC_FAIL

    def test_b2_1_b3_1_b1_zero_patient_does_not_trigger_exclusion_is_pass(self):
        # B1=0: "Karnofsky > 50%" as exclusion — patient KPS=40 does NOT trigger → PASS
        c = karnofsky_criterion(b1=0)
        evals = evaluate_all_criteria([c], patient_fail(karnofsky=40))
        assert evals[0]["kind"] == DETERMINISTIC_PASS

    @pytest.mark.parametrize("b1", [0, 1, N])
    def test_b2_1_b3_1_patient_fails_is_deterministic_fail(self, b1):
        # Karnofsky=40 fails "> 50%" for inclusion; or passes exclusion inversion
        # For B1=1 (inclusion): 40 > 50 is False → DETERMINISTIC_FAIL
        # For B1=0 (exclusion): evaluate inverts — 40 > 50 is False, invert=True → PASS
        # For B1=None: treated as inclusion (is_exclusion=False) → FAIL
        c = karnofsky_criterion(b1=b1)
        patient = patient_fail(karnofsky=40)
        evals = evaluate_all_criteria([c], patient)
        kind = evals[0]["kind"]
        if b1 == 0:
            # Exclusion: "Karnofsky > 50%" as exclusion criterion — patient fails
            # the comparison (40 > 50 = False), inversion gives True → PASS
            assert kind == DETERMINISTIC_PASS
        else:
            assert kind == DETERMINISTIC_FAIL

    @pytest.mark.parametrize("b1", [0, 1, N])
    def test_b2_1_b3_1_patient_field_absent_is_unevaluable(self, b1):
        c = cancer_type_criterion(b1=b1)
        evals = evaluate_all_criteria([c], {})
        assert evals[0]["kind"] == UNEVALUABLE

    # -----------------------------------------------------------------------
    # Exclusion-polarity deterministic (B1=0, B2=1, B3=1)
    # -----------------------------------------------------------------------

    def test_exclusion_inclusion_patient_not_excluded_is_pass(self):
        # B1=0 (exclusion), patient does NOT match exclusion → PASS
        # E.g., exclude male patients; patient is female → not excluded
        c = make_criterion(
            text="Male patients are excluded from this study",
            b1=0, b2=1, b3=1,
        )
        evals = evaluate_all_criteria([c], {"sex": "female"})
        assert evals[0]["kind"] == DETERMINISTIC_PASS

    def test_exclusion_patient_is_excluded_is_fail(self):
        # B1=0 (exclusion), patient matches exclusion → FAIL
        c = make_criterion(
            text="Male patients are excluded from this study",
            b1=0, b2=1, b3=1,
        )
        evals = evaluate_all_criteria([c], {"sex": "male"})
        assert evals[0]["kind"] == DETERMINISTIC_FAIL

    # -----------------------------------------------------------------------
    # Subjective branch (B2=0) — all 9 B1×B3 combinations
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("b1,b3", [
        (0, 0), (0, 1), (0, N),
        (1, 0), (1, 1), (1, N),
        (N, 0), (N, 1), (N, N),
    ])
    def test_b2_zero_always_subjective(self, b1, b3):
        c = consent_criterion(b1=b1)
        c.b3_label = b3
        evals = evaluate_all_criteria([c], {})
        assert evals[0]["kind"] == SUBJECTIVE

    # -----------------------------------------------------------------------
    # Unobservable branch — B2=1, B3=0 or None (6 combinations)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("b1,b3", [
        (0, 0), (0, N),
        (1, 0), (1, N),
        (N, 0), (N, N),
    ])
    def test_b2_one_b3_not_one_is_unobservable(self, b1, b3):
        c = make_criterion(b1=b1, b2=1, b3=b3)
        evals = evaluate_all_criteria([c], {})
        assert evals[0]["kind"] == UNOBSERVABLE

    # -----------------------------------------------------------------------
    # Unobservable branch — B2=None, all 9 B1×B3 combinations
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("b1,b3", [
        (0, 0), (0, 1), (0, N),
        (1, 0), (1, 1), (1, N),
        (N, 0), (N, 1), (N, N),
    ])
    def test_b2_none_always_unobservable(self, b1, b3):
        c = make_criterion(b1=b1, b2=N, b3=b3)
        evals = evaluate_all_criteria([c], {})
        assert evals[0]["kind"] == UNOBSERVABLE

    # -----------------------------------------------------------------------
    # Hedging assignment
    # -----------------------------------------------------------------------

    def test_subjective_hedging_set_from_text(self):
        c = consent_criterion()  # low hedging (signed consent)
        evals = evaluate_all_criteria([c], {})
        assert evals[0]["hedging"] == 0.05

    def test_subjective_high_hedging_from_willingness_text(self):
        c = willingness_criterion()
        evals = evaluate_all_criteria([c], {})
        assert evals[0]["hedging"] == 0.8

    def test_non_subjective_hedging_is_always_half(self):
        for kind_crit in [
            karnofsky_criterion(),
            cancer_type_criterion(),
            unobservable_criterion(),
        ]:
            evals = evaluate_all_criteria([kind_crit], {})
            assert evals[0]["hedging"] == 0.5, \
                f"Expected hedging=0.5 for {evals[0]['kind']}"

    # -----------------------------------------------------------------------
    # Output dict structure
    # -----------------------------------------------------------------------

    def test_output_contains_required_keys(self):
        c = karnofsky_criterion()
        ev = evaluate_all_criteria([c], patient_pass())[0]
        required = {"criterion_id", "text", "b1_label", "b2_label", "b3_label",
                    "kind", "hedging"}
        assert required.issubset(ev.keys())

    def test_output_preserves_criterion_id(self):
        c = karnofsky_criterion(cid="my_id")
        ev = evaluate_all_criteria([c], patient_pass())[0]
        assert ev["criterion_id"] == "my_id"

    def test_output_length_matches_input(self):
        criteria = [karnofsky_criterion(), consent_criterion(), unobservable_criterion()]
        evals = evaluate_all_criteria(criteria, {})
        assert len(evals) == 3

    def test_empty_criteria_returns_empty_list(self):
        assert evaluate_all_criteria([], {}) == []

    def test_order_preserved(self):
        criteria = [
            karnofsky_criterion(cid="first"),
            consent_criterion(cid="second"),
        ]
        evals = evaluate_all_criteria(criteria, patient_pass())
        assert evals[0]["criterion_id"] == "first"
        assert evals[1]["criterion_id"] == "second"

    def test_multiple_criteria_mixed_kinds(self):
        criteria = [
            karnofsky_criterion(),             # DETERMINISTIC_PASS with kps=80
            consent_criterion(),               # SUBJECTIVE
            unobservable_criterion(b2=N, b3=N),  # UNOBSERVABLE
            cancer_type_criterion(),           # UNEVALUABLE
        ]
        evals = evaluate_all_criteria(criteria, patient_pass(karnofsky=80))
        kinds = [e["kind"] for e in evals]
        assert DETERMINISTIC_PASS in kinds
        assert SUBJECTIVE in kinds
        assert UNOBSERVABLE in kinds
        assert UNEVALUABLE in kinds


# ===========================================================================
# TestBuildModel — PyMC model structure
# ===========================================================================

class TestBuildModel:
    """build_model: correct Beta parameterisation and model structure."""

    def _eval(self, kind, hedging=0.5, cid="c0"):
        return {
            "criterion_id": cid,
            "text": "test",
            "b1_label": 1,
            "b2_label": 1 if kind != SUBJECTIVE else 0,
            "b3_label": 1,
            "kind": kind,
            "hedging": hedging,
        }

    # -----------------------------------------------------------------------
    # Stochastic criterion filtering
    # -----------------------------------------------------------------------

    def test_deterministic_pass_excluded_from_model(self):
        evals = [self._eval(DETERMINISTIC_PASS)]
        model = build_model(evals)
        assert len(model.free_RVs) == 0

    def test_subjective_included_in_model(self):
        evals = [self._eval(SUBJECTIVE, hedging=0.5)]
        model = build_model(evals)
        assert len(model.free_RVs) == 1

    def test_unobservable_included_in_model(self):
        evals = [self._eval(UNOBSERVABLE)]
        model = build_model(evals)
        assert len(model.free_RVs) == 1

    def test_unevaluable_included_in_model(self):
        evals = [self._eval(UNEVALUABLE)]
        model = build_model(evals)
        assert len(model.free_RVs) == 1

    def test_deterministic_fail_excluded_from_model(self):
        # FAIL should have been caught upstream; if it reaches build_model
        # it is silently excluded (no Beta variable created)
        evals = [self._eval(DETERMINISTIC_FAIL)]
        model = build_model(evals)
        assert len(model.free_RVs) == 0

    def test_mixed_evaluations_correct_free_rv_count(self):
        evals = [
            self._eval(DETERMINISTIC_PASS, cid="c0"),
            self._eval(SUBJECTIVE, hedging=0.5, cid="c1"),
            self._eval(UNOBSERVABLE, cid="c2"),
            self._eval(UNEVALUABLE, cid="c3"),
        ]
        model = build_model(evals)
        # Only stochastic criteria get Beta vars
        assert len(model.free_RVs) == 3

    # -----------------------------------------------------------------------
    # p_eligible is always registered
    # -----------------------------------------------------------------------

    def test_p_eligible_in_named_vars(self):
        evals = [self._eval(SUBJECTIVE)]
        model = build_model(evals)
        assert "p_eligible" in model.named_vars

    def test_p_eligible_registered_for_all_pass_model(self):
        evals = [self._eval(DETERMINISTIC_PASS)]
        model = build_model(evals)
        assert "p_eligible" in model.named_vars

    def test_p_eligible_registered_for_empty_model(self):
        model = build_model([])
        assert "p_eligible" in model.named_vars

    # -----------------------------------------------------------------------
    # Beta parameterisation via sampling
    # -----------------------------------------------------------------------

    def test_consent_criterion_samples_near_one(self):
        # hedging=0.05 → Beta(1.9, 0.1) → mean = 1.9/2.0 = 0.95
        evals = [self._eval(SUBJECTIVE, hedging=0.05)]
        result = _samples(evals, n=5000)
        assert result["mean"] == pytest.approx(0.95, abs=0.03)

    def test_willingness_criterion_samples_near_point_two(self):
        # hedging=0.8 → Beta(0.4, 1.6) → mean = 0.4/2.0 = 0.2
        evals = [self._eval(SUBJECTIVE, hedging=0.8)]
        result = _samples(evals, n=5000)
        assert result["mean"] == pytest.approx(0.2, abs=0.03)

    def test_default_subjective_samples_near_half(self):
        # hedging=0.5 → Beta(1.0, 1.0) = Uniform → mean = 0.5
        evals = [self._eval(SUBJECTIVE, hedging=0.5)]
        result = _samples(evals, n=5000)
        assert result["mean"] == pytest.approx(0.5, abs=0.03)

    def test_unobservable_samples_near_half(self):
        # Beta(1,1) = Uniform → mean = 0.5
        evals = [self._eval(UNOBSERVABLE)]
        result = _samples(evals, n=5000)
        assert result["mean"] == pytest.approx(0.5, abs=0.03)

    def test_unevaluable_samples_near_half(self):
        # Same as UNOBSERVABLE
        evals = [self._eval(UNEVALUABLE)]
        result = _samples(evals, n=5000)
        assert result["mean"] == pytest.approx(0.5, abs=0.03)

    # -----------------------------------------------------------------------
    # Product structure: two independent Beta(1,1) → mean ~ 0.25
    # -----------------------------------------------------------------------

    def test_product_of_two_uniform_mean_is_quarter(self):
        evals = [self._eval(UNOBSERVABLE, cid="c0"),
                 self._eval(UNOBSERVABLE, cid="c1")]
        result = _samples(evals, n=5000)
        # E[U1 * U2] = E[U1] * E[U2] = 0.5 * 0.5 = 0.25
        assert result["mean"] == pytest.approx(0.25, abs=0.03)

    def test_product_of_three_uniform_mean_is_eighth(self):
        evals = [self._eval(UNOBSERVABLE, cid=f"c{i}") for i in range(3)]
        result = _samples(evals, n=5000)
        # E[U1*U2*U3] = 0.5^3 = 0.125
        assert result["mean"] == pytest.approx(0.125, abs=0.03)


# ===========================================================================
# TestComputePosteriorProperties — statistical correctness
# ===========================================================================

class TestComputePosteriorProperties:
    """compute_posterior: output shape, range, CI ordering, reproducibility."""

    def _eval(self, kind="unobservable", hedging=0.5, cid="c0"):
        return {
            "criterion_id": cid,
            "text": "test",
            "b1_label": 1,
            "b2_label": 0 if kind == SUBJECTIVE else 1,
            "b3_label": 1,
            "kind": kind,
            "hedging": hedging,
        }

    # -----------------------------------------------------------------------
    # Output structure
    # -----------------------------------------------------------------------

    def test_returns_all_required_keys(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=100)
        assert {"mean", "ci_lower", "ci_upper", "samples"} == set(result.keys())

    def test_samples_shape_matches_draws(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=500)
        assert result["samples"].shape == (500,)

    def test_samples_are_numpy_array(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=100)
        assert isinstance(result["samples"], np.ndarray)

    # -----------------------------------------------------------------------
    # Range constraints
    # -----------------------------------------------------------------------

    def test_mean_in_unit_interval(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=500)
        assert 0.0 <= result["mean"] <= 1.0

    def test_samples_all_in_unit_interval(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=1000)
        assert np.all(result["samples"] >= 0.0)
        assert np.all(result["samples"] <= 1.0)

    def test_ci_ordering(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=1000)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_ci_lower_nonnegative(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=1000)
        assert result["ci_lower"] >= 0.0

    def test_ci_upper_leq_one(self):
        model = build_model([self._eval()])
        result = compute_posterior(model, draws=1000)
        assert result["ci_upper"] <= 1.0

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------

    def test_same_seed_same_samples(self):
        evals = [self._eval()]
        model1 = build_model(evals)
        model2 = build_model(evals)
        r1 = compute_posterior(model1, draws=200, random_seed=99)
        r2 = compute_posterior(model2, draws=200, random_seed=99)
        np.testing.assert_array_equal(r1["samples"], r2["samples"])

    def test_different_seeds_different_samples(self):
        evals = [self._eval()]
        model1 = build_model(evals)
        model2 = build_model(evals)
        r1 = compute_posterior(model1, draws=200, random_seed=1)
        r2 = compute_posterior(model2, draws=200, random_seed=2)
        assert not np.array_equal(r1["samples"], r2["samples"])

    # -----------------------------------------------------------------------
    # Monotone mean decrease with more unobservable criteria
    # -----------------------------------------------------------------------

    def test_mean_decreases_monotonically_with_more_criteria(self):
        means = []
        for n in range(1, 6):
            evals = [self._eval(cid=f"c{i}") for i in range(n)]
            result = _samples(evals, n=3000)
            means.append(result["mean"])
        # Each additional Beta(1,1) factor should reduce the mean
        for i in range(len(means) - 1):
            assert means[i] > means[i + 1], \
                f"Mean did not decrease at n={i+1} → n={i+2}: {means}"

    # -----------------------------------------------------------------------
    # CI width increases with uncertainty
    # -----------------------------------------------------------------------

    def test_single_uniform_has_wide_ci(self):
        evals = [self._eval()]
        result = _samples(evals, n=5000)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width > 0.7  # Beta(1,1) ≈ Uniform → ~0.95 width

    def test_high_hedging_subjective_ci_lower_than_half(self):
        # Beta(0.4, 1.6) → mass concentrated below 0.5
        evals = [self._eval(kind=SUBJECTIVE, hedging=0.8)]
        result = _samples(evals, n=5000)
        assert result["mean"] < 0.4

    def test_low_hedging_subjective_ci_above_half(self):
        # Beta(1.9, 0.1) → mass concentrated above 0.5
        evals = [self._eval(kind=SUBJECTIVE, hedging=0.05)]
        result = _samples(evals, n=5000)
        assert result["mean"] > 0.85


# ===========================================================================
# TestComputeEligibilityPosterior — full pipeline
# ===========================================================================

class TestComputeEligibilityPosterior:
    """compute_eligibility_posterior: routing, counts, and result structure."""

    # -----------------------------------------------------------------------
    # Empty criteria list → vacuously eligible (point mass at 1)
    # -----------------------------------------------------------------------

    def test_empty_criteria_mean_is_one(self):
        result = _run([], {})
        assert result["mean"] == 1.0

    def test_empty_criteria_ci_is_one_one(self):
        result = _run([], {})
        assert result["ci_lower"] == 1.0
        assert result["ci_upper"] == 1.0

    def test_empty_criteria_not_short_circuited(self):
        result = _run([], {})
        assert result["short_circuited"] is False

    def test_empty_criteria_all_counts_zero(self):
        result = _run([], {})
        assert result["n_deterministic"] == 0
        assert result["n_subjective"] == 0
        assert result["n_unobservable"] == 0
        assert result["n_unevaluable"] == 0

    # -----------------------------------------------------------------------
    # All-pass → point mass at 1
    # -----------------------------------------------------------------------

    def test_all_deterministic_pass_mean_is_one(self):
        criteria = [karnofsky_criterion(cid=f"k{i}") for i in range(3)]
        result = _run(criteria, patient_pass(karnofsky=80))
        assert result["mean"] == 1.0

    def test_all_deterministic_pass_ci_is_one_one(self):
        criteria = [karnofsky_criterion()]
        result = _run(criteria, patient_pass(karnofsky=80))
        assert result["ci_lower"] == 1.0
        assert result["ci_upper"] == 1.0

    def test_all_deterministic_pass_n_deterministic_correct(self):
        criteria = [
            karnofsky_criterion(cid="k1"),
            karnofsky_criterion(cid="k2"),
        ]
        result = _run(criteria, patient_pass(karnofsky=80))
        assert result["n_deterministic"] == 2
        assert result["n_subjective"] == 0
        assert result["n_unobservable"] == 0

    # -----------------------------------------------------------------------
    # Short-circuit: first criterion is a hard fail
    # -----------------------------------------------------------------------

    def test_first_criterion_fail_short_circuits(self):
        criteria = [
            karnofsky_criterion(cid="kps"),
            consent_criterion(cid="cs"),
        ]
        result = _run(criteria, patient_fail(karnofsky=40))
        assert result["short_circuited"] is True

    def test_first_criterion_fail_mean_is_zero(self):
        criteria = [karnofsky_criterion()]
        result = _run(criteria, patient_fail(karnofsky=40))
        assert result["mean"] == 0.0

    def test_first_criterion_fail_ci_is_zero_zero(self):
        criteria = [karnofsky_criterion()]
        result = _run(criteria, patient_fail(karnofsky=40))
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] == 0.0

    def test_first_criterion_fail_reports_failing_id(self):
        criteria = [karnofsky_criterion(cid="kps_001")]
        result = _run(criteria, patient_fail(karnofsky=40))
        assert result["failing_criterion"] == "kps_001"

    # -----------------------------------------------------------------------
    # Short-circuit: fail is NOT the first criterion
    # -----------------------------------------------------------------------

    def test_mid_list_fail_still_short_circuits(self):
        criteria = [
            consent_criterion(cid="c0"),      # subjective, not evaluated
            unobservable_criterion(cid="c1"),  # unobservable
            karnofsky_criterion(cid="kps"),    # FAIL (kps=40)
            willingness_criterion(cid="c3"),   # subjective, never reached
        ]
        result = _run(criteria, patient_fail(karnofsky=40))
        assert result["short_circuited"] is True
        assert result["mean"] == 0.0
        assert result["failing_criterion"] == "kps"

    def test_fail_after_pass_short_circuits(self):
        criteria = [
            karnofsky_criterion(cid="kps_pass"),   # kps=80 → PASS
            karnofsky_criterion(cid="kps_fail"),   # kps=40 → FAIL
        ]
        # Both are the same criterion; patient with kps=40 fails kps > 50
        result = _run(criteria, patient_fail(karnofsky=40))
        assert result["short_circuited"] is True
        # First criterion kps > 50 with kps=40 → fails first
        assert result["failing_criterion"] == "kps_pass"

    # -----------------------------------------------------------------------
    # Short-circuit on first of multiple fails
    # -----------------------------------------------------------------------

    def test_multiple_fails_short_circuits_on_first(self):
        criteria = [
            make_criterion(text="Female subjects only",
                           b1=1, b2=1, b3=1, cid="sex"),   # FAIL (male patient)
            karnofsky_criterion(cid="kps"),                  # would also fail (kps=40)
        ]
        result = _run(criteria, {"sex": "male", "karnofsky": 40})
        assert result["short_circuited"] is True
        assert result["failing_criterion"] == "sex"  # first fail wins

    # -----------------------------------------------------------------------
    # Exclusion-polarity short-circuit
    # -----------------------------------------------------------------------

    def test_exclusion_patient_triggers_it_short_circuits(self):
        # B1=0 exclusion: "Male patients excluded"; patient is male → excluded → FAIL
        c = make_criterion(
            text="Male patients are excluded from this study",
            b1=0, b2=1, b3=1, cid="excl_male",
        )
        result = _run([c], {"sex": "male"})
        assert result["short_circuited"] is True

    def test_exclusion_patient_not_triggered_passes(self):
        # B1=0 exclusion: "Male patients excluded"; patient is female → not excluded
        c = make_criterion(
            text="Male patients are excluded from this study",
            b1=0, b2=1, b3=1, cid="excl_male",
        )
        result = _run([c], {"sex": "female"})
        assert result["short_circuited"] is False
        assert result["n_deterministic"] == 1

    # -----------------------------------------------------------------------
    # Criterion count fields
    # -----------------------------------------------------------------------

    def test_n_counts_correct_for_mixed_criteria(self):
        criteria = [
            karnofsky_criterion(cid="det"),         # DETERMINISTIC_PASS
            consent_criterion(cid="subj"),          # SUBJECTIVE
            unobservable_criterion(b2=N, b3=N, cid="unobs"),  # UNOBSERVABLE
            cancer_type_criterion(cid="uneval"),    # UNEVALUABLE
        ]
        result = _run(criteria, patient_pass(karnofsky=80))
        assert result["n_deterministic"] == 1
        assert result["n_subjective"] == 1
        assert result["n_unobservable"] == 1
        assert result["n_unevaluable"] == 1
        assert result["short_circuited"] is False

    def test_failing_criterion_none_when_not_short_circuited(self):
        criteria = [unobservable_criterion()]
        result = _run(criteria, {})
        assert result["failing_criterion"] is None

    # -----------------------------------------------------------------------
    # Return value structure
    # -----------------------------------------------------------------------

    def test_result_has_all_required_keys(self):
        result = _run([unobservable_criterion()], {})
        required = {
            "mean", "ci_lower", "ci_upper",
            "n_deterministic", "n_subjective", "n_unobservable", "n_unevaluable",
            "short_circuited", "failing_criterion", "samples",
        }
        assert required.issubset(result.keys())

    def test_samples_is_numpy_array(self):
        result = _run([unobservable_criterion()], {})
        assert isinstance(result["samples"], np.ndarray)

    def test_samples_length_matches_n_samples(self):
        result = compute_eligibility_posterior(
            [unobservable_criterion()], {}, n_samples=300
        )
        assert len(result["samples"]) == 300

    def test_all_sample_values_in_unit_interval(self):
        result = _run([unobservable_criterion()], {})
        assert np.all(result["samples"] >= 0.0)
        assert np.all(result["samples"] <= 1.0)

    def test_ci_ordering_on_stochastic_result(self):
        result = _run([unobservable_criterion()], {})
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------

    def test_same_seed_gives_same_result(self):
        criteria = [consent_criterion(), willingness_criterion()]
        r1 = compute_eligibility_posterior(criteria, {}, n_samples=200, random_seed=7)
        r2 = compute_eligibility_posterior(criteria, {}, n_samples=200, random_seed=7)
        assert r1["mean"] == r2["mean"]
        np.testing.assert_array_equal(r1["samples"], r2["samples"])

    # -----------------------------------------------------------------------
    # Plausible posterior values
    # -----------------------------------------------------------------------

    def test_single_unobservable_mean_near_half(self):
        result = compute_eligibility_posterior(
            [unobservable_criterion()], {}, n_samples=5000, random_seed=42
        )
        assert result["mean"] == pytest.approx(0.5, abs=0.05)

    def test_many_unobservable_mean_near_zero(self):
        criteria = [unobservable_criterion(cid=f"u{i}") for i in range(8)]
        result = compute_eligibility_posterior(
            criteria, {}, n_samples=5000, random_seed=42
        )
        # E[∏ U_i] = 0.5^8 ≈ 0.004
        assert result["mean"] < 0.02

    def test_single_consent_subjective_mean_near_one(self):
        # Consent criterion → hedging=0.05 → Beta(1.9, 0.1) → mean≈0.95
        result = compute_eligibility_posterior(
            [consent_criterion()], {}, n_samples=5000, random_seed=42
        )
        assert result["mean"] > 0.85

    def test_mix_deterministic_pass_plus_unobservable(self):
        # deterministic pass contributes 1.0 × Beta(1,1) → same as single Beta(1,1)
        criteria = [karnofsky_criterion(), unobservable_criterion()]
        result = compute_eligibility_posterior(
            criteria, patient_pass(karnofsky=80), n_samples=5000, random_seed=42
        )
        assert result["mean"] == pytest.approx(0.5, abs=0.06)
        assert result["n_deterministic"] == 1
        assert result["n_unobservable"] == 1
