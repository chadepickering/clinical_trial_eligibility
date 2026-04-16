"""
Ingestion runner — fetches oncology trials from ClinicalTrials.gov and writes
them to the local DuckDB store.

Usage:
    python ingest.py                        # full run (15,000 trials)
    python ingest.py --max-trials 100       # small test run
    python ingest.py --max-trials 100 --db data/processed/trials.duckdb
"""

import argparse
import sys

from ingestion.api_client import ClinicalTrialsClient
from ingestion.database import get_connection, init_schema
from ingestion.parser import parse_study


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest oncology trials from ClinicalTrials.gov")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=15000,
        help="Maximum number of trials to fetch (default: 15000)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/processed/trials.duckdb",
        help="Path to DuckDB file (default: data/processed/trials.duckdb)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Connecting to database: {args.db}")
    conn = get_connection(args.db)
    init_schema(conn)

    client = ClinicalTrialsClient(max_trials=args.max_trials)

    inserted = 0
    skipped = 0
    errors = 0

    print(f"Fetching up to {args.max_trials} trials...")

    for raw in client.fetch():
        try:
            row = parse_study(raw)

            if not row["nct_id"]:
                skipped += 1
                continue

            conn.execute("""
                INSERT OR IGNORE INTO trials (
                    nct_id, brief_title, conditions, interventions,
                    intervention_types, intervention_descriptions,
                    intervention_other_names, phases, status,
                    min_age, max_age, sex, std_ages,
                    primary_outcomes, secondary_outcomes,
                    brief_summary, detailed_description, eligibility_text,
                    mesh_conditions, mesh_interventions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row["nct_id"],
                row["brief_title"],
                row["conditions"],
                row["interventions"],
                row["intervention_types"],
                row["intervention_descriptions"],
                row["intervention_other_names"],
                row["phases"],
                row["status"],
                row["min_age"],
                row["max_age"],
                row["sex"],
                row["std_ages"],
                row["primary_outcomes"],
                row["secondary_outcomes"],
                row["brief_summary"],
                row["detailed_description"],
                row["eligibility_text"],
                row["mesh_conditions"],
                row["mesh_interventions"],
            ])

            inserted += 1

            if inserted % 100 == 0:
                print(f"  {inserted} inserted, {skipped} skipped, {errors} errors...")

        except Exception as e:
            errors += 1
            nct_id = raw.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "unknown")
            print(f"  ERROR on {nct_id}: {e}", file=sys.stderr)

    conn.close()

    print()
    print("Ingestion complete.")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped}  (duplicate or missing nct_id)")
    print(f"  Errors   : {errors}")


if __name__ == "__main__":
    main()
