"""
Credible interval computation and uncertainty reporting.

Summarizes PyMC posterior traces into human-readable uncertainty outputs
for the Streamlit interface and API responses.
"""
import arviz as az


def summarize_posterior(trace) -> dict:
    """
    Returns HDI, mean, and a textual uncertainty tier:
      "high confidence" | "moderate uncertainty" | "high uncertainty"
    """
    pass
