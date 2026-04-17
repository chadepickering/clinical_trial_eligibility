"""
DuckDB connection and schema management.

Manages two stores:
  - trials: trial metadata and unprocessed criteria text
  - criteria: labeled criterion objects post-NLP classification
"""

import duckdb


def get_connection(path: str = "data/processed/trials.duckdb") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(path)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            nct_id               VARCHAR PRIMARY KEY,
            brief_title          VARCHAR,
            conditions           VARCHAR[],
            interventions        VARCHAR[],
            intervention_types   VARCHAR[],    -- parallel array to interventions (DRUG, DEVICE, etc.)
            phases               VARCHAR[],
            status               VARCHAR,
            min_age              VARCHAR,      -- stored as string e.g. "18 Years" — parsed downstream
            max_age              VARCHAR,      -- nullable — often absent
            sex                  VARCHAR,      -- "ALL", "MALE", "FEMALE"
            std_ages             VARCHAR[],
            primary_outcomes          VARCHAR[],
            secondary_outcomes        VARCHAR[],    -- secondary endpoint measures — NER source for LAB_VALUE, SCALE, THRESHOLD, TIMEFRAME
            intervention_descriptions VARCHAR[],    -- parallel to interventions — dosing/route/schedule prose — NER source for DRUG, THRESHOLD, TIMEFRAME
            intervention_other_names  VARCHAR[],    -- parallel to interventions — drug aliases e.g. "anti-PD1" — NER source for DRUG
            brief_summary             TEXT,
            detailed_description TEXT,         -- nullable
            eligibility_text     TEXT,
            mesh_conditions      VARCHAR[],    -- MeSH-normalized condition terms
            mesh_interventions   VARCHAR[],    -- MeSH-normalized drug/intervention terms
            ingested_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS criteria (
            criterion_id           VARCHAR PRIMARY KEY,  -- composed as nct_id + '_' + position index
            nct_id                 VARCHAR REFERENCES trials(nct_id),
            text                   TEXT,
            section                VARCHAR,      -- "inclusion", "exclusion", "unknown"
            position               INTEGER,      -- 0-based index within trial
            -- B1/B2/B3 weak labels (populated by NLP layer; NULL = no signal / ambiguous)
            b1_label               INTEGER,      -- 1=inclusion, 0=exclusion
            b2_label               INTEGER,      -- 1=objective, 0=subjective
            b3_label               INTEGER,      -- 1=observable, 0=unobservable
            b2_confidence          FLOAT,        -- 0.0–1.0 signal strength
            b3_confidence          FLOAT,        -- 0.0–1.0 signal strength
            -- NER outputs (populated by NER layer)
            extracted_conditions   VARCHAR[],
            extracted_drugs        VARCHAR[],
            extracted_lab_values   VARCHAR[],
            extracted_thresholds   VARCHAR[],
            extracted_demographics VARCHAR[],
            extracted_scales       VARCHAR[],    -- e.g. "ECOG 0-2", "CTCAE Grade 3"
            extracted_timeframes   VARCHAR[],    -- e.g. "within 6 months"
            processed_at           TIMESTAMP
        )
    """)
