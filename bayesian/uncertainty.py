"""
Credible interval computation and uncertainty reporting for the Bayesian
eligibility model.

Provides two public functions:

    summarize_posterior(posterior_result, hdi_prob) → dict
        Accepts the dict returned by compute_eligibility_posterior() and
        produces a human-readable summary with an arviz HDI, an uncertainty
        tier, and a plain-English explanation.

    uncertainty_decomposition(posterior_result) → dict
        Breaks down the criterion counts by kind and returns the dominant
        source of uncertainty (deterministic / subjective / unobservable /
        unevaluable / disqualified).

Uncertainty tiers are defined by the 95 % HDI width:

    "high confidence"      HDI width < 0.20  — narrow, well-constrained
    "moderate uncertainty" HDI width < 0.50  — mixed deterministic+stochastic
    "high uncertainty"     HDI width ≥ 0.50  — mostly stochastic or missing data

Short-circuited results (P = 0 exactly) are labeled "disqualified" regardless
of HDI width.

Usage:
    from bayesian.eligibility_model import compute_eligibility_posterior
    from bayesian.uncertainty import summarize_posterior, uncertainty_decomposition

    result = compute_eligibility_posterior(criteria, patient)
    summary = summarize_posterior(result)
    decomp  = uncertainty_decomposition(result)
"""

import numpy as np
import arviz as az


# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE_WIDTH   = 0.20
_MODERATE_UNCERTAINTY_WIDTH = 0.50

# HDI probability used throughout this module
DEFAULT_HDI_PROB = 0.95


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_posterior(
    posterior_result: dict,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> dict:
    """
    Summarize a posterior result from compute_eligibility_posterior().

    Args:
        posterior_result: dict returned by compute_eligibility_posterior().
                          Must contain keys: mean, ci_lower, ci_upper,
                          short_circuited, samples (np.ndarray).
        hdi_prob:         HDI probability mass (default 0.95).

    Returns dict with keys:
        mean              — posterior mean P(eligible)
        hdi_lower         — lower bound of arviz HDI
        hdi_upper         — upper bound of arviz HDI
        hdi_width         — hdi_upper - hdi_lower
        hdi_prob          — probability mass used (e.g. 0.95)
        tier              — "disqualified" | "high confidence" |
                            "moderate uncertainty" | "high uncertainty"
        short_circuited   — bool (True → hard disqualifier found)
        failing_criterion — criterion_id of hard disqualifier, or None
        explanation       — human-readable one-sentence interpretation
    """
    short_circuited   = posterior_result.get("short_circuited", False)
    failing_criterion = posterior_result.get("failing_criterion")
    mean_p            = float(posterior_result["mean"])
    samples           = np.asarray(posterior_result["samples"]).flatten()

    # Compute arviz HDI
    hdi = az.hdi(samples, hdi_prob=hdi_prob)
    hdi_lower = float(hdi[0])
    hdi_upper = float(hdi[1])
    hdi_width = hdi_upper - hdi_lower

    # Tier classification
    if short_circuited:
        tier = "disqualified"
    elif hdi_width < _HIGH_CONFIDENCE_WIDTH:
        tier = "high confidence"
    elif hdi_width < _MODERATE_UNCERTAINTY_WIDTH:
        tier = "moderate uncertainty"
    else:
        tier = "high uncertainty"

    explanation = _explain(tier, mean_p, hdi_lower, hdi_upper, failing_criterion)

    return {
        "mean":              mean_p,
        "hdi_lower":         hdi_lower,
        "hdi_upper":         hdi_upper,
        "hdi_width":         hdi_width,
        "hdi_prob":          hdi_prob,
        "tier":              tier,
        "short_circuited":   short_circuited,
        "failing_criterion": failing_criterion,
        "explanation":       explanation,
    }


def uncertainty_decomposition(posterior_result: dict) -> dict:
    """
    Break down criterion counts by kind and identify the dominant uncertainty
    source.

    Args:
        posterior_result: dict returned by compute_eligibility_posterior().

    Returns dict with keys:
        n_deterministic   — criteria resolved as hard passes
        n_subjective      — criteria with hedging-shaped Beta prior
        n_unobservable    — criteria with Beta(1,1) prior (unobservable or unlabeled)
        n_unevaluable     — objective+observable but patient field absent
        n_total_stochastic — sum of subjective + unobservable + unevaluable
        short_circuited   — bool
        dominant_source   — "disqualified" | "subjective" | "unobservable" |
                            "unevaluable" | "deterministic" | "none"
        dominant_count    — count associated with dominant_source
    """
    short_circuited  = posterior_result.get("short_circuited", False)
    n_deterministic  = int(posterior_result.get("n_deterministic",  0))
    n_subjective     = int(posterior_result.get("n_subjective",     0))
    n_unobservable   = int(posterior_result.get("n_unobservable",   0))
    n_unevaluable    = int(posterior_result.get("n_unevaluable",    0))
    n_total_stochastic = n_subjective + n_unobservable + n_unevaluable

    if short_circuited:
        dominant_source = "disqualified"
        dominant_count  = 0
    elif n_total_stochastic == 0:
        dominant_source = "deterministic"
        dominant_count  = n_deterministic
    else:
        # Largest stochastic category wins
        stochastic_counts = {
            "subjective":   n_subjective,
            "unobservable": n_unobservable,
            "unevaluable":  n_unevaluable,
        }
        dominant_source = max(stochastic_counts, key=stochastic_counts.get)
        dominant_count  = stochastic_counts[dominant_source]
        if dominant_count == 0:
            dominant_source = "none"

    return {
        "n_deterministic":    n_deterministic,
        "n_subjective":       n_subjective,
        "n_unobservable":     n_unobservable,
        "n_unevaluable":      n_unevaluable,
        "n_total_stochastic": n_total_stochastic,
        "short_circuited":    short_circuited,
        "dominant_source":    dominant_source,
        "dominant_count":     dominant_count,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _explain(
    tier: str,
    mean_p: float,
    hdi_lower: float,
    hdi_upper: float,
    failing_criterion: str | None,
) -> str:
    """Generate a one-sentence plain-English explanation for the given tier."""
    if tier == "disqualified":
        crit_str = f" ({failing_criterion})" if failing_criterion else ""
        return (
            f"The patient is definitively ineligible due to a hard "
            f"disqualifying criterion{crit_str}."
        )
    pct = round(mean_p * 100, 1)
    lo  = round(hdi_lower * 100, 1)
    hi  = round(hdi_upper * 100, 1)
    if tier == "high confidence":
        return (
            f"Eligibility probability is {pct}% (95% HDI: {lo}%–{hi}%), "
            f"with a narrow credible interval indicating high confidence."
        )
    if tier == "moderate uncertainty":
        return (
            f"Eligibility probability is {pct}% (95% HDI: {lo}%–{hi}%), "
            f"with moderate uncertainty from partially observable criteria."
        )
    # high uncertainty
    return (
        f"Eligibility probability is {pct}% (95% HDI: {lo}%–{hi}%), "
        f"with high uncertainty due to many unobservable or unevaluable criteria."
    )
