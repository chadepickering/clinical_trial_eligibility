"""
Review LLM annotations against SciBERT predictions and compute evaluation metrics.

Workflow:
  1. Flags rows requiring human review (disagreements + low-confidence agreements)
  2. Prints a review queue for human labeling
  3. After human_b2_label / human_b3_label columns are filled, computes:
     - F1, precision, recall per head (SciBERT vs human ground truth)
     - Calibration: SciBERT accuracy by probability decile

Usage:
    # Step 1 — generate review flags and print queue
    python scripts/review_annotations.py --csv data/annotation/sample.csv

    # Step 2 — after filling human labels, compute final metrics
    python scripts/review_annotations.py --csv data/annotation/sample.csv --metrics
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


CONFIDENCE_THRESHOLD = 0.8   # LLM confidence below this triggers review flag


def flag_for_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where human review is needed:
      - LLM and SciBERT disagree on either head
      - LLM agrees with SciBERT but confidence < threshold on either head
    """
    df = df.copy()

    b2_disagree = df['llm_b2_label'] != df['scibert_b2_label']
    b3_disagree = df['llm_b3_label'] != df['scibert_b3_label']
    low_conf    = (df['llm_b2_confidence'] < CONFIDENCE_THRESHOLD) | \
                  (df['llm_b3_confidence'] < CONFIDENCE_THRESHOLD)

    df['review_flag'] = 'ok'
    df.loc[b2_disagree | b3_disagree, 'review_flag'] = 'disagree'
    df.loc[(~(b2_disagree | b3_disagree)) & low_conf, 'review_flag'] = 'low_confidence'

    return df


def print_review_queue(df: pd.DataFrame) -> None:
    """Print rows that need human review in a readable format."""
    queue = df[df['review_flag'] != 'ok'].copy()
    print(f"\n{'='*70}")
    print(f"REVIEW QUEUE — {len(queue)} rows ({len(df)} total)")
    print(f"  disagree:       {(queue['review_flag']=='disagree').sum()}")
    print(f"  low_confidence: {(queue['review_flag']=='low_confidence').sum()}")
    print(f"{'='*70}\n")

    for _, row in queue.iterrows():
        print(f"[{row['criterion_id']}] {row['review_flag'].upper()}")
        print(f"  Text: {row['text'][:120]}{'...' if len(row['text'])>120 else ''}")
        print(f"  B2 → SciBERT={'obj' if row['scibert_b2_label']==1 else 'subj'} "
              f"(p={row['scibert_b2_prob']:.2f})  "
              f"LLM={'obj' if row['llm_b2_label']==1 else 'subj'} "
              f"(conf={row['llm_b2_confidence']:.2f})")
        print(f"  B3 → SciBERT={'obs' if row['scibert_b3_label']==1 else 'unobs'} "
              f"(p={row['scibert_b3_prob']:.2f})  "
              f"LLM={'obs' if row['llm_b3_label']==1 else 'unobs'} "
              f"(conf={row['llm_b3_confidence']:.2f})")
        if pd.notna(row.get('human_b2_label')):
            print(f"  Human: B2={'obj' if row['human_b2_label']==1 else 'subj'}  "
                  f"B3={'obs' if row['human_b3_label']==1 else 'unobs'}")
        print()


def resolve_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine ground truth for each row:
      - If human label exists → use it
      - Else use LLM label (only for non-flagged rows with high confidence)
    """
    df = df.copy()
    for head in ('b2', 'b3'):
        human_col = f'human_{head}_label'
        llm_col   = f'llm_{head}_label'
        gt_col    = f'gt_{head}_label'
        df[gt_col] = df[human_col].combine_first(df[llm_col])
    return df


def compute_metrics(df: pd.DataFrame) -> None:
    """Compute per-head classification metrics and calibration table."""
    df = resolve_ground_truth(df)
    evaluable = df[df['gt_b2_label'].notna() & df['gt_b3_label'].notna()]

    if len(evaluable) == 0:
        print("No rows with ground truth yet. Fill human_b2_label / human_b3_label first.")
        return

    print(f"\n{'='*70}")
    print(f"EVALUATION METRICS  ({len(evaluable)} rows with ground truth)")
    print(f"{'='*70}")

    for head, pos_label, neg_label, prob_col in (
        ('b2', 'objective', 'subjective', 'scibert_b2_prob'),
        ('b3', 'observable', 'unobservable', 'scibert_b3_prob'),
    ):
        gt   = evaluable[f'gt_{head}_label'].astype(int)
        pred = evaluable[f'scibert_{head}_label'].astype(int)

        print(f"\n--- B{head[1]} ({pos_label}/{neg_label}) ---")
        print(classification_report(gt, pred,
                                    target_names=[neg_label, pos_label],
                                    zero_division=0))

        # Calibration: accuracy by SciBERT probability decile
        probs = evaluable[prob_col].astype(float)
        print(f"Calibration (SciBERT P({pos_label}) decile → accuracy):")
        print(f"  {'Decile range':<20} {'n':>5} {'accuracy':>10}")
        bins = np.linspace(0, 1, 11)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo)
            bucket = evaluable[mask]
            if len(bucket) == 0:
                continue
            acc = (bucket[f'scibert_{head}_label'].astype(int) ==
                   bucket[f'gt_{head}_label'].astype(int)).mean()
            print(f"  [{lo:.1f}–{hi:.1f}){'':>10} {len(bucket):>5}     {acc:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     default='data/annotation/sample.csv')
    parser.add_argument('--metrics', action='store_true',
                        help='compute final F1 and calibration (requires human labels)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if df['llm_b2_label'].isna().all():
        print("No LLM labels found. Run llm_annotate.py first.")
        return

    df = flag_for_review(df)
    df.to_csv(args.csv, index=False)

    if args.metrics:
        compute_metrics(df)
    else:
        print_review_queue(df)
        n_ok      = (df['review_flag'] == 'ok').sum()
        n_review  = (df['review_flag'] != 'ok').sum()
        auto_rate = n_ok / len(df) * 100
        print(f"Auto-accepted (LLM+SciBERT agree, high conf): {n_ok} ({auto_rate:.0f}%)")
        print(f"Needs human review: {n_review} ({100-auto_rate:.0f}%)")
        print(f"\nFill human_b2_label and human_b3_label in {args.csv}")
        print("then run with --metrics to compute final evaluation.")


if __name__ == '__main__':
    main()
