"""
Regex/heuristic weak supervision for bootstrap labeling.

Generates noisy labels for B1/B2/B3 classification tasks
to support training before manual annotation is available.

B1: Inclusion vs Exclusion (derived from section header)
B2: Objective vs Subjective (keyword patterns — "ECOG", "must have", "adequate", etc.)
B3: Observable vs Unobservable (lab values, vitals = observable; history = unobservable)
"""


def label_criterion(criterion: dict) -> dict:
    """
    Adds weak label fields: b1_label, b2_label, b3_label, confidence.
    """
    pass
