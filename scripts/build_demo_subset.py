"""
Build a demo-sized subset of the trial database for Streamlit Community Cloud.

Selects 500 oncology trials with rich eligibility criteria, stratified by
cancer type for variety. NCT00127920 (the pipeline walkthrough trial) is
always included.

Output:
    data/demo/trials.duckdb   — subset DuckDB (~10MB)
    data/demo/chroma/         — subset ChromaDB (~30MB)

Usage:
    python scripts/build_demo_subset.py
    python scripts/build_demo_subset.py --db data/processed/trials.duckdb \\
                                         --chroma data/processed/chroma \\
                                         --out-dir data/demo \\
                                         --n 500 --min-criteria 15

The app reads DATA_DIR env var (defaults to data/processed).
Set DATA_DIR=data/demo for the cloud deployment.
"""

import argparse
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ANCHOR_NCT = "NCT00127920"  # pipeline walkthrough trial — always included

ONCOLOGY_TERMS = [
    "cancer", "carcinoma", "tumor", "tumour", "leukemia", "leukaemia",
    "lymphoma", "neoplasm", "sarcoma", "melanoma", "myeloma", "glioma",
    "blastoma", "adenocarcinoma", "mesothelioma",
]

# Simplified cancer type labels for stratification — ordered most-specific first
_TYPE_PATTERNS = [
    ("ovarian",       ["ovarian", "ovary"]),
    ("breast",        ["breast"]),
    ("lung",          ["lung", "non-small cell", "small cell", "nsclc", "sclc"]),
    ("colorectal",    ["colorectal", "colon", "rectal"]),
    ("prostate",      ["prostate"]),
    ("hematologic",   ["leukemia", "leukaemia", "lymphoma", "myeloma", "aml", "cll", "cml"]),
    ("bladder",       ["bladder", "urothelial"]),
    ("pancreatic",    ["pancreatic", "pancreas"]),
    ("gastric",       ["gastric", "stomach", "esophageal", "oesophageal"]),
    ("head_neck",     ["head and neck", "head & neck", "squamous cell"]),
    ("liver",         ["hepatocellular", "liver", "cholangiocarcinoma", "biliary"]),
    ("kidney",        ["renal", "kidney"]),
    ("brain",         ["glioma", "glioblastoma", "brain", "cns", "meningioma"]),
    ("cervical",      ["cervical", "cervix"]),
    ("endometrial",   ["endometrial", "uterine", "uterus"]),
    ("sarcoma",       ["sarcoma"]),
    ("melanoma",      ["melanoma"]),
    ("thyroid",       ["thyroid"]),
    ("other",         []),  # catch-all
]


def _cancer_type(conditions: list[str]) -> str:
    text = " ".join(conditions).lower() if conditions else ""
    for label, keywords in _TYPE_PATTERNS:
        if label == "other":
            return "other"
        if any(kw in text for kw in keywords):
            return label
    return "other"


def _is_oncology(conditions: list[str], mesh_conditions: list[str]) -> bool:
    text = " ".join((conditions or []) + (mesh_conditions or [])).lower()
    return any(term in text for term in ONCOLOGY_TERMS)


def select_trials(db_path: str, n: int, min_criteria: int, seed: int) -> list[str]:
    """Return a list of nct_ids: anchor + stratified sample of oncology trials.

    Stratification is two-dimensional:
      1. Cancer type  — so the demo covers a broad disease landscape
      2. Criteria-count bucket (5-9 / 10-19 / 20+) — so the demo includes
         simple trials (easy to pass the 30% coverage gate and trigger the
         Bayesian model) as well as complex trials (many unobservable criteria).
         Roughly 50% of slots go to the 5-19 range to maximise demos where
         the Bayesian output is visible.
    """
    import duckdb

    con = duckdb.connect(db_path, read_only=True)

    rows = con.execute("""
        SELECT t.nct_id,
               COUNT(c.criterion_id) AS n_crit,
               t.conditions,
               t.mesh_conditions
        FROM trials t
        JOIN criteria c ON t.nct_id = c.nct_id
        GROUP BY t.nct_id, t.conditions, t.mesh_conditions
        HAVING COUNT(c.criterion_id) >= ?
        ORDER BY COUNT(c.criterion_id) DESC
    """, [min_criteria]).fetchall()
    con.close()

    candidates = [
        (nct_id, n_crit, conds or [], mesh or [])
        for nct_id, n_crit, conds, mesh in rows
        if _is_oncology(conds or [], mesh or [])
    ]

    print(f"Oncology candidates with ≥{min_criteria} criteria: {len(candidates):,}")

    # Always include anchor
    selected = {ANCHOR_NCT}
    non_anchor = [(nct, nc, conds, mesh) for nct, nc, conds, mesh in candidates
                  if nct != ANCHOR_NCT]

    def _crit_bucket(n_crit: int) -> str:
        if n_crit < 10:  return "5-9"
        if n_crit < 20:  return "10-19"
        return "20+"

    # Build a (cancer_type, crit_bucket) → [nct_id, ...] index
    cell: dict[tuple[str, str], list[str]] = {}
    for nct, nc, conds, mesh in non_anchor:
        key = (_cancer_type(conds), _crit_bucket(nc))
        cell.setdefault(key, []).append(nct)

    # Desired bucket weights: favour the simpler tiers for better demo coverage
    bucket_weight = {"5-9": 0.35, "10-19": 0.35, "20+": 0.30}
    n_remaining = n - len(selected)

    rng = random.Random(seed)
    pool: list[str] = []

    # Two-pass fill: first pass allocates proportionally across (type, bucket) cells
    n_types  = len({_cancer_type(row[2]) for row in non_anchor})
    per_type = max(1, n_remaining // n_types)

    for cancer_type in sorted({_cancer_type(row[2]) for row in non_anchor}):
        for bucket, weight in bucket_weight.items():
            key = (cancer_type, bucket)
            if key not in cell:
                continue
            quota = max(1, round(per_type * weight))
            choices = cell[key][:]
            rng.shuffle(choices)
            pool.extend(choices[:quota])

    # Second pass: fill remaining slots randomly from unused candidates
    already = set(pool) | selected
    remaining = [nct for nct, _, _, _ in non_anchor if nct not in already]
    rng.shuffle(remaining)
    pool.extend(remaining[:n_remaining - len(pool)])

    final = list(selected) + pool[:n - len(selected)]
    rng.shuffle(final)

    # Summary
    type_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for nct_id in final:
        row = next((r for r in candidates if r[0] == nct_id), None)
        if row:
            type_counts[_cancer_type(row[2])] = type_counts.get(_cancer_type(row[2]), 0) + 1
            bucket_counts[_crit_bucket(row[1])] = bucket_counts.get(_crit_bucket(row[1]), 0) + 1

    print(f"\nSelected {len(final)} trials by cancer type:")
    for label, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {cnt}")
    print(f"\nBy criteria-count bucket:")
    for b in ("5-9", "10-19", "20+"):
        print(f"  {b:<10} {bucket_counts.get(b, 0)}")
    print(f"\n  Anchor: {ANCHOR_NCT}")

    return final


def export_duckdb(src_path: str, out_path: str, nct_ids: list[str]) -> None:
    import duckdb

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    placeholders = ", ".join(f"'{nct}'" for nct in nct_ids)

    src = duckdb.connect(src_path, read_only=True)
    trials_df   = src.execute(f"SELECT * FROM trials   WHERE nct_id IN ({placeholders})").df()
    criteria_df = src.execute(f"SELECT * FROM criteria WHERE nct_id IN ({placeholders})").df()
    src.close()

    dst = duckdb.connect(out_path)
    dst.execute("CREATE TABLE trials   AS SELECT * FROM trials_df")
    dst.execute("CREATE TABLE criteria AS SELECT * FROM criteria_df")

    t_count = dst.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    c_count = dst.execute("SELECT COUNT(*) FROM criteria").fetchone()[0]
    dst.close()

    print(f"\nDuckDB exported: {t_count} trials, {c_count} criteria → {out_path}")


def export_chroma(src_chroma: str, out_chroma: str, nct_ids: list[str]) -> None:
    from rag.vector_store import get_client, get_collection, upsert_trials

    if os.path.exists(out_chroma):
        shutil.rmtree(out_chroma)
    os.makedirs(out_chroma, exist_ok=True)

    # Read from source
    src_client = get_client(src_chroma)
    src_col = get_collection(src_client)

    print(f"\nFetching {len(nct_ids)} embeddings from source ChromaDB...")
    result = src_col.get(
        ids=nct_ids,
        include=["embeddings", "documents", "metadatas"],
    )

    found_ids = result["ids"]
    missing = set(nct_ids) - set(found_ids)
    if missing:
        print(f"  Warning: {len(missing)} trials not found in ChromaDB (will re-embed): "
              f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Write to destination
    dst_client = get_client(out_chroma)
    dst_col = get_collection(dst_client)
    upsert_trials(
        dst_col,
        result["ids"],
        result["embeddings"],
        result["documents"],
        result["metadatas"],
    )
    print(f"ChromaDB exported: {len(result['ids'])} trials → {out_chroma}")

    # Re-embed any missing trials from DuckDB
    if missing:
        print(f"\nRe-embedding {len(missing)} missing trials...")
        _reembed_missing(list(missing), out_chroma)


def _reembed_missing(nct_ids: list[str], chroma_dir: str) -> None:
    import duckdb
    from rag.embedder import build_corpus, embed_corpus
    from rag.vector_store import get_client, get_collection, upsert_trials

    db_path = os.path.join(ROOT, "data", "processed", "trials.duckdb")
    placeholders = ", ".join(f"'{n}'" for n in nct_ids)
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(f"""
        SELECT nct_id, brief_title, brief_summary, eligibility_text,
               conditions, phases, status, sex, min_age, max_age, std_ages
        FROM trials WHERE nct_id IN ({placeholders})
    """).fetchall()
    con.close()

    cols = ["nct_id", "brief_title", "brief_summary", "eligibility_text",
            "conditions", "phases", "status", "sex", "min_age", "max_age", "std_ages"]
    trial_dicts = [dict(zip(cols, r)) for r in rows]

    texts, metadatas, ids = build_corpus(trial_dicts)
    embeddings = embed_corpus(texts, show_progress=True)

    client = get_client(chroma_dir)
    col = get_collection(client)
    upsert_trials(col, ids, embeddings, texts, metadatas)
    print(f"Re-embedded {len(ids)} trials.")


def main():
    parser = argparse.ArgumentParser(description="Build demo subset for cloud deployment")
    parser.add_argument("--db",            default="data/processed/trials.duckdb")
    parser.add_argument("--chroma",        default="data/processed/chroma")
    parser.add_argument("--out-dir",       default="data/demo")
    parser.add_argument("--n",             type=int, default=1000)
    parser.add_argument("--min-criteria",  type=int, default=5)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    db_path     = os.path.join(ROOT, args.db)
    chroma_path = os.path.join(ROOT, args.chroma)
    out_dir     = os.path.join(ROOT, args.out_dir)
    out_db      = os.path.join(out_dir, "trials.duckdb")
    out_chroma  = os.path.join(out_dir, "chroma")

    print(f"Building demo subset: {args.n} trials, ≥{args.min_criteria} criteria")
    print(f"Anchor trial: {ANCHOR_NCT}\n")

    t0 = time.time()

    nct_ids = select_trials(db_path, args.n, args.min_criteria, args.seed)
    export_duckdb(db_path, out_db, nct_ids)
    export_chroma(chroma_path, out_chroma, nct_ids)

    elapsed = time.time() - t0

    # Report file sizes
    db_mb = os.path.getsize(out_db) / 1e6
    chroma_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(out_chroma)
        for f in files
    ) / 1e6
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  {out_db}: {db_mb:.1f} MB")
    print(f"  {out_chroma}/: {chroma_mb:.1f} MB")
    print(f"\nNext: commit data/demo/ to git, then deploy to Streamlit Community Cloud.")
    print(f"Set DATA_DIR=data/demo in Streamlit secrets.")


if __name__ == "__main__":
    main()
