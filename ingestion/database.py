"""
DuckDB connection and schema management.

Manages two stores:
  - raw_trials: trial metadata and unprocessed criteria text
  - criteria: labeled criterion objects post-NLP classification
"""


def get_connection(path: str = "data/processed/trials.duckdb"):
    pass


def init_schema(conn) -> None:
    pass
