"""
Clinical named entity recognition using a SciBERT NER head.

Extracts structured entities from criterion text:
  - conditions (cancer types, diagnoses)
  - drugs / interventions
  - lab values and thresholds (e.g., "ANC >= 1500/uL")
  - demographics (age, sex)
"""


def extract_entities(criterion_text: str) -> dict:
    """
    Returns dict of entity lists keyed by entity type.
    """
    pass
