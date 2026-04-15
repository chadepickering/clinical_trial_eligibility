"""
JSON → structured DuckDB records.

Extracts key fields from raw ClinicalTrials.gov API responses:
nctId, eligibilityCriteria, minimumAge, maximumAge, conditions,
interventions, phases, overallStatus, primaryOutcomes.
"""


def parse_study(raw: dict) -> dict:
    pass
