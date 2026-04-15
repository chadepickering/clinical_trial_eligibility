"""
Splits a raw eligibility criteria blob into individual criterion sentences.

The ClinicalTrials.gov criteria field is a single free-text block mixing
inclusion and exclusion criteria under headers. This module segments it
into a list of discrete, classifiable criterion strings.
"""


def split_criteria(raw_text: str) -> list[dict]:
    """
    Returns a list of dicts: {"text": str, "section": "inclusion"|"exclusion"|"unknown"}
    """
    pass
