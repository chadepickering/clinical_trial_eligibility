"""
Patient-vs-criterion matching.

For each labeled criterion, compares patient profile fields against
extracted entities (lab thresholds, conditions, demographics) to produce:
  - match_result: True / False / None (for unobservable criteria)
  - confidence: float in [0, 1]
  - evidence: extracted comparison details
"""


def evaluate_criterion(criterion: dict, patient: dict) -> dict:
    """
    Returns: {
        "criterion_id": str,
        "b2_label": "objective" | "subjective",
        "b3_label": "observable" | "unobservable",
        "match_result": bool | None,
        "confidence": float,
        "evidence": dict
    }
    """
    pass
