"""
PyMC Bayesian eligibility model.

Computes posterior P(eligible | patient, trial) as a product over all criteria:

    P(eligible) = ∏ P(meets criterion_i)

Where each factor is determined by the criterion's B2/B3 labels:

    B2=Objective + B3=Observable → deterministic 0 or 1 (via criterion_evaluator)
    B2=Subjective                → Beta(alpha, beta) prior shaped by hedging strength
    B3=Unobservable              → Beta(1, 1) uninformative prior (marginalized)
    B2/B3=None (unlabeled)       → Beta(1, 1) uninformative prior (conservative)
    Objective+Observable but
      patient field unknown      → Beta(1, 1) (cannot evaluate, treat as uncertain)

Design note — prior predictive sampling:
    This model has no observed data: we compute a prior predictive distribution
    over P(eligible) given the criterion structure. pm.sample_prior_predictive()
    is used rather than pm.sample() (NUTS) because:
      - There is no likelihood to condition on
      - Prior predictive sampling is ~100× faster for this use case
      - Results are mathematically equivalent: both yield samples from the
        joint distribution of Beta priors propagated through the product

Public API:
    evaluate_all_criteria(criteria, patient)  → list[dict]
    build_model(criteria_evaluations)         → pm.Model
    compute_posterior(model, draws)           → dict
    compute_eligibility_posterior(...)        → dict   # convenience wrapper

Usage:
    from bayesian.criterion_evaluator import load_criteria_for_trial
    from bayesian.eligibility_model import compute_eligibility_posterior
    import duckdb

    con = duckdb.connect("data/processed/trials.duckdb")
    criteria = load_criteria_for_trial("NCT00127920", con)
    patient = {"age": 52, "sex": "female", "karnofsky": 80, ...}
    result = compute_eligibility_posterior(criteria, patient)
    # result: {mean, ci_lower, ci_upper, n_deterministic, n_subjective,
    #          n_unobservable, n_unevaluable, short_circuited, samples}
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Suppress PyMC sampling log messages (e.g. "Sampling: [p_0, p_1, ...]")
# that appear during pm.sample_prior_predictive(). Warnings are preserved.
logging.getLogger("pymc").setLevel(logging.WARNING)

from bayesian.criterion_evaluator import (
    Criterion,
    estimate_hedging,
    evaluate_objective_criterion,
)

# ---------------------------------------------------------------------------
# Criterion kind constants
# ---------------------------------------------------------------------------

DETERMINISTIC_PASS = "deterministic_pass"   # obj+obs, patient meets it
DETERMINISTIC_FAIL = "deterministic_fail"   # obj+obs, patient fails → short-circuit
SUBJECTIVE         = "subjective"           # b2=0 → Beta prior (hedging-shaped)
UNOBSERVABLE       = "unobservable"         # b3=0 → Beta(1,1) uninformative
UNEVALUABLE        = "unevaluable"          # obj+obs, patient field absent → Beta(1,1)

# ---------------------------------------------------------------------------
# Step 1 — classify every criterion against the patient profile
# ---------------------------------------------------------------------------


def evaluate_all_criteria(
    criteria: list[Criterion],
    patient_profile: dict,
) -> list[dict]:
    """
    Classify each criterion into one of five kinds and collect hedging strength.

    Returns a list of evaluation dicts, one per criterion, preserving order.
    Each dict has keys:
        criterion_id  — str
        text          — str
        b1_label      — int | None
        b2_label      — int | None
        b3_label      — int | None
        kind          — one of the five DETERMINISTIC_PASS / ... constants
        hedging       — float (relevant only for SUBJECTIVE; 0.5 for others)

    No exceptions are raised — unevaluable criteria gracefully degrade to
    Beta(1,1) treatment in build_model.
    """
    evaluations = []

    for c in criteria:
        is_obj = c.b2_label == 1
        is_obs = c.b3_label == 1

        if is_obj and is_obs:
            match = evaluate_objective_criterion(c, patient_profile)
            if match is True:
                kind = DETERMINISTIC_PASS
            elif match is False:
                kind = DETERMINISTIC_FAIL
            else:
                # Objective+observable but patient profile lacks the field
                kind = UNEVALUABLE
        elif c.b2_label == 0:
            kind = SUBJECTIVE
        else:
            # b3_label=0 (unobservable), or either label is None (unknown)
            kind = UNOBSERVABLE

        evaluations.append(
            {
                "criterion_id": c.criterion_id,
                "text": c.text,
                "b1_label": c.b1_label,
                "b2_label": c.b2_label,
                "b3_label": c.b3_label,
                "kind": kind,
                "hedging": estimate_hedging(c.text) if kind == SUBJECTIVE else 0.5,
            }
        )

    return evaluations


# ---------------------------------------------------------------------------
# Step 2 — build PyMC model from stochastic criteria only
# ---------------------------------------------------------------------------


def build_model(criteria_evaluations: list[dict]) -> pm.Model:
    """
    Build a PyMC model containing one Beta variable per stochastic criterion.

    Deterministic passes are excluded (they contribute factor 1.0 trivially).
    Deterministic fails must be handled upstream — if any are present the
    model will still run but the caller should have short-circuited already.

    Beta parameterisation:
        SUBJECTIVE:   alpha = 2*(1-hedging),  beta = 2*hedging
                      hedging=0.05 → Beta(1.9, 0.1) — very likely met (consent)
                      hedging=0.50 → Beta(1.0, 1.0) — maximum uncertainty
                      hedging=0.80 → Beta(0.4, 1.6) — skewed toward uncertain
        UNOBSERVABLE: Beta(1, 1) — uniform, no information
        UNEVALUABLE:  Beta(1, 1) — uniform, no information

    The overall eligibility probability is the product of all Beta variables,
    registered as a pm.Deterministic so it appears in prior predictive samples.

    Args:
        criteria_evaluations: output of evaluate_all_criteria()

    Returns:
        pm.Model ready for pm.sample_prior_predictive()
    """
    stochastic = [
        e for e in criteria_evaluations
        if e["kind"] in (SUBJECTIVE, UNOBSERVABLE, UNEVALUABLE)
    ]

    with pm.Model() as model:
        beta_vars = []

        for i, ev in enumerate(stochastic):
            if ev["kind"] == SUBJECTIVE:
                hedging = float(ev["hedging"])
                alpha = max(2.0 * (1.0 - hedging), 0.1)
                beta_param = max(2.0 * hedging, 0.1)
            else:
                alpha, beta_param = 1.0, 1.0

            beta_vars.append(
                pm.Beta(f"p_{i}", alpha=alpha, beta=beta_param)
            )

        if beta_vars:
            pm.Deterministic("p_eligible", pt.prod(pt.stack(beta_vars)))
        else:
            # All criteria were deterministic passes — register constant 1.0
            pm.Deterministic("p_eligible", pt.as_tensor_variable(1.0))

    return model


# ---------------------------------------------------------------------------
# Step 3 — draw prior predictive samples
# ---------------------------------------------------------------------------


def compute_posterior(
    model: pm.Model,
    draws: int = 2000,
    random_seed: int = 42,
) -> dict:
    """
    Draw prior predictive samples of p_eligible from the model.

    Uses pm.sample_prior_predictive rather than NUTS because there is no
    likelihood: we are propagating Beta prior uncertainty through the product.

    Args:
        model:       pm.Model built by build_model()
        draws:       number of samples (default 2000)
        random_seed: for reproducibility (default 42)

    Returns:
        dict with keys: mean, ci_lower, ci_upper, samples (np.ndarray)
    """
    with model:
        idata = pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
        )

    samples = idata.prior["p_eligible"].values.flatten()

    return {
        "mean":     float(np.mean(samples)),
        "ci_lower": float(np.percentile(samples, 2.5)),
        "ci_upper": float(np.percentile(samples, 97.5)),
        "samples":  samples,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper — full pipeline in one call
# ---------------------------------------------------------------------------

def _zero_result(
    criteria_evaluations: list[dict],
    failing_criterion_id: str,
) -> dict:
    """Return a short-circuited result for a deterministic disqualifier."""
    n_subjective   = sum(1 for e in criteria_evaluations if e["kind"] == SUBJECTIVE)
    n_unobservable = sum(1 for e in criteria_evaluations if e["kind"] == UNOBSERVABLE)
    n_unevaluable  = sum(1 for e in criteria_evaluations if e["kind"] == UNEVALUABLE)
    n_deterministic = sum(
        1 for e in criteria_evaluations if e["kind"] == DETERMINISTIC_PASS
    )
    return {
        "mean":               0.0,
        "ci_lower":           0.0,
        "ci_upper":           0.0,
        "n_deterministic":    n_deterministic,
        "n_subjective":       n_subjective,
        "n_unobservable":     n_unobservable,
        "n_unevaluable":      n_unevaluable,
        "short_circuited":    True,
        "failing_criterion":  failing_criterion_id,
        "samples":            np.zeros(1),
    }


def _point_mass_result(
    criteria_evaluations: list[dict],
    n_samples: int,
) -> dict:
    """Return a point-mass result when all criteria are deterministic passes."""
    n_deterministic = sum(
        1 for e in criteria_evaluations if e["kind"] == DETERMINISTIC_PASS
    )
    return {
        "mean":              1.0,
        "ci_lower":          1.0,
        "ci_upper":          1.0,
        "n_deterministic":   n_deterministic,
        "n_subjective":      0,
        "n_unobservable":    0,
        "n_unevaluable":     0,
        "short_circuited":   False,
        "failing_criterion": None,
        "samples":           np.ones(n_samples),
    }


def compute_eligibility_posterior(
    criteria: list[Criterion],
    patient_profile: dict,
    n_samples: int = 2000,
    random_seed: int = 42,
) -> dict:
    """
    Full pipeline: classify criteria → short-circuit check → sample posterior.

    Args:
        criteria:        list of Criterion objects from load_criteria_for_trial()
        patient_profile: patient dict (see criterion_evaluator.py for schema)
        n_samples:       prior predictive draws (default 2000)
        random_seed:     for reproducibility (default 42)

    Returns:
        dict with keys:
            mean              — posterior mean P(eligible)
            ci_lower          — 2.5th percentile
            ci_upper          — 97.5th percentile
            n_deterministic   — criteria resolved deterministically (pass)
            n_subjective      — criteria with Beta(hedging) prior
            n_unobservable    — criteria with Beta(1,1) prior (unobservable/unknown)
            n_unevaluable     — objective+observable but patient field absent
            short_circuited   — True if a hard disqualifier was found
            failing_criterion — criterion_id of first hard fail, else None
            samples           — np.ndarray of posterior samples
    """
    # Classify all criteria
    evaluations = evaluate_all_criteria(criteria, patient_profile)

    # Short-circuit: first deterministic fail → P(eligible) = 0 exactly
    for ev in evaluations:
        if ev["kind"] == DETERMINISTIC_FAIL:
            return _zero_result(evaluations, ev["criterion_id"])

    # All-deterministic-pass edge case → P(eligible) = 1 exactly
    stochastic = [e for e in evaluations if e["kind"] in
                  (SUBJECTIVE, UNOBSERVABLE, UNEVALUABLE)]
    if not stochastic:
        return _point_mass_result(evaluations, n_samples)

    # Build model and sample
    model = build_model(evaluations)
    posterior = compute_posterior(model, draws=n_samples, random_seed=random_seed)

    n_deterministic = sum(1 for e in evaluations if e["kind"] == DETERMINISTIC_PASS)
    n_subjective    = sum(1 for e in evaluations if e["kind"] == SUBJECTIVE)
    n_unobservable  = sum(1 for e in evaluations if e["kind"] == UNOBSERVABLE)
    n_unevaluable   = sum(1 for e in evaluations if e["kind"] == UNEVALUABLE)

    return {
        "mean":              posterior["mean"],
        "ci_lower":          posterior["ci_lower"],
        "ci_upper":          posterior["ci_upper"],
        "n_deterministic":   n_deterministic,
        "n_subjective":      n_subjective,
        "n_unobservable":    n_unobservable,
        "n_unevaluable":     n_unevaluable,
        "short_circuited":   False,
        "failing_criterion": None,
        "samples":           posterior["samples"],
    }
