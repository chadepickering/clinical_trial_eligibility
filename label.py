"""
Labeling runner — splits eligibility criteria text and applies weak labels,
writing the results to the criteria table in the local DuckDB store.

Processes all trials in the database by default. Already-labeled trials are
skipped unless --reprocess is passed, which deletes and rebuilds their rows.

Usage:
    python label.py                            # label all unlabeled trials
    python label.py --reprocess                # reprocess all trials
    python label.py --db data/processed/trials.duckdb
"""

import argparse
import sys
from datetime import datetime, timezone

from ingestion.database import get_connection
from nlp.criterion_splitter import split_criteria
from nlp.weak_labeler import label_criterion


def parse_args():
    parser = argparse.ArgumentParser(description="Split and weakly label eligibility criteria")
    parser.add_argument(
        "--db",
        type=str,
        default="data/processed/trials.duckdb",
        help="Path to DuckDB file (default: data/processed/trials.duckdb)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Delete existing criteria rows and reprocess all trials",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Connecting to database: {args.db}")
    conn = get_connection(args.db)

    # Fetch all trials, or only those without existing criteria rows
    if args.reprocess:
        print("--reprocess: clearing existing criteria rows...")
        conn.execute("DELETE FROM criteria")
        rows = conn.execute("SELECT nct_id, eligibility_text FROM trials").fetchall()
    else:
        rows = conn.execute("""
            SELECT t.nct_id, t.eligibility_text
            FROM trials t
            WHERE NOT EXISTS (
                SELECT 1 FROM criteria c WHERE c.nct_id = t.nct_id
            )
        """).fetchall()

    total_trials = len(rows)
    print(f"Trials to process: {total_trials}")

    processed = 0
    skipped = 0
    total_criteria = 0
    errors = 0
    now = datetime.now(timezone.utc)

    for nct_id, eligibility_text in rows:
        try:
            criteria = split_criteria(eligibility_text)

            if not criteria:
                skipped += 1
                continue

            for criterion in criteria:
                labeled = label_criterion(criterion)
                criterion_id = f"{nct_id}_{labeled['position']}"

                conn.execute("""
                    INSERT OR IGNORE INTO criteria (
                        criterion_id, nct_id, text, section, position,
                        b1_label, b2_label, b3_label,
                        b2_confidence, b3_confidence,
                        processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    criterion_id,
                    nct_id,
                    labeled["text"],
                    labeled["section"],
                    labeled["position"],
                    labeled["b1_label"],
                    labeled["b2_label"],
                    labeled["b3_label"],
                    labeled["b2_confidence"],
                    labeled["b3_confidence"],
                    now,
                ])
                total_criteria += 1

            processed += 1

            if processed % 100 == 0:
                print(f"  {processed}/{total_trials} trials, {total_criteria} criteria so far...")

        except Exception as e:
            errors += 1
            print(f"  ERROR on {nct_id}: {e}", file=sys.stderr)

    conn.close()

    print()
    print("Labeling complete.")
    print(f"  Trials processed : {processed}")
    print(f"  Trials skipped   : {skipped}  (no parseable criteria)")
    print(f"  Criteria written : {total_criteria}")
    print(f"  Errors           : {errors}")


if __name__ == "__main__":
    main()
