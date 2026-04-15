"""
PyMC Bayesian eligibility model.

Computes posterior P(eligible | patient, trial) integrating:
  - Deterministic pass/fail for objective + observable criteria
  - Beta prior for subjective criteria (uncertainty over physician judgment)
  - Marginalization over unobservable criteria (treat as latent variables)

Output: posterior samples → mean probability + 95% credible interval.
"""
import pymc as pm


def build_model(criteria_evaluations: list[dict]) -> pm.Model:
    pass


def compute_posterior(model: pm.Model, draws: int = 2000) -> dict:
    """
    Returns: {"mean": float, "ci_lower": float, "ci_upper": float, "trace": ...}
    """
    pass
