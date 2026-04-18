"""
Mine the most discriminative unigrams and bigrams from the weakly-labeled
criteria corpus for B2 (objective/subjective) and B3 (observable/unobservable).

Uses TF-IDF mean ratio: average TF-IDF weight in class A divided by average
in class B. High ratio = term is proportionally more common in class A than B.

Only uses labeled rows (where label IS NOT NULL) — None rows are excluded.

Usage:
    python scripts/mine_discriminative_ngrams.py
    python scripts/mine_discriminative_ngrams.py --db data/processed/trials.duckdb --top 50
"""

import argparse
import duckdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def top_discriminative(texts, labels, pos_label, neg_label, top_n, min_df=20):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                          max_features=10000, lowercase=True)
    X = vec.fit_transform(texts)
    names = np.array(vec.get_feature_names_out())

    pos_mask = np.array(labels) == pos_label
    neg_mask = np.array(labels) == neg_label

    eps = 1e-9
    pos_mean = X[pos_mask].mean(axis=0).A1
    neg_mean = X[neg_mask].mean(axis=0).A1

    pos_disc = pos_mean / (neg_mean + eps)
    neg_disc = neg_mean / (pos_mean + eps)

    return (
        names[np.argsort(pos_disc)[::-1][:top_n]],
        names[np.argsort(neg_disc)[::-1][:top_n]],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',  default='data/processed/trials.duckdb')
    parser.add_argument('--top', type=int, default=40)
    args = parser.parse_args()

    conn = duckdb.connect(args.db)

    # ── B2 ───────────────────────────────────────────────────────────────────
    rows = conn.execute(
        "SELECT text, b2_label FROM criteria WHERE b2_label IS NOT NULL"
    ).fetchall()
    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    top_obj, top_subj = top_discriminative(texts, labels, pos_label=1,
                                           neg_label=0, top_n=args.top)
    print('=== B2 TOP OBJECTIVE (1) NGRAMS ===')
    for t in top_obj:  print(f'  {t}')
    print()
    print('=== B2 TOP SUBJECTIVE (0) NGRAMS ===')
    for t in top_subj: print(f'  {t}')
    print()

    # ── B3 ───────────────────────────────────────────────────────────────────
    rows = conn.execute(
        "SELECT text, b3_label FROM criteria WHERE b3_label IS NOT NULL"
    ).fetchall()
    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    top_obs, top_unobs = top_discriminative(texts, labels, pos_label=1,
                                            neg_label=0, top_n=args.top)
    print('=== B3 TOP OBSERVABLE (1) NGRAMS ===')
    for t in top_obs:   print(f'  {t}')
    print()
    print('=== B3 TOP UNOBSERVABLE (0) NGRAMS ===')
    for t in top_unobs: print(f'  {t}')


if __name__ == '__main__':
    main()
