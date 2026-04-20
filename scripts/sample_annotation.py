"""
Sample 500 criteria from None rows for manual annotation.

Selects criteria where the weak labeler abstained on B2 or B3 (or both),
runs the trained SciBERT checkpoint to get predictions and probabilities,
and writes a CSV ready for LLM annotation.

Stratifies by section (inclusion/exclusion/unknown) proportionally so the
sample reflects the real corpus distribution.

Usage:
    python scripts/sample_annotation.py
    python scripts/sample_annotation.py --db data/processed/trials.duckdb \
        --checkpoint nlp/checkpoints/best_model.pt --n 500 --out data/annotation/sample.csv
"""

import argparse
import os
import random

import duckdb
import torch
import pandas as pd
from torch.utils.data import DataLoader

from nlp.multitask_classifier import CriterionClassifier, CriterionDataset, load_tokenizer


def load_none_rows(db_path: str) -> list[dict]:
    """Load criteria where b2_label OR b3_label is NULL."""
    conn = duckdb.connect(db_path)
    rows = conn.execute("""
        SELECT criterion_id, nct_id, text, section,
               b2_label, b3_label, b2_confidence, b3_confidence
        FROM criteria
        WHERE (b2_label IS NULL OR b3_label IS NULL)
          AND text IS NOT NULL
          AND length(text) > 0
    """).fetchall()
    conn.close()
    cols = ['criterion_id', 'nct_id', 'text', 'section',
            'b2_label', 'b3_label', 'b2_confidence', 'b3_confidence']
    return [dict(zip(cols, r)) for r in rows]


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """
    Proportional stratified sample by section.
    Ensures the sample mirrors the None-row corpus section distribution.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['section']].append(r)

    total = len(rows)
    sample = []
    rng = random.Random(seed)

    for section, bucket in buckets.items():
        k = max(1, round(n * len(bucket) / total))
        sample.extend(rng.sample(bucket, min(k, len(bucket))))

    # Trim or top-up to exactly n
    rng.shuffle(sample)
    if len(sample) > n:
        sample = sample[:n]
    elif len(sample) < n:
        remaining = [r for r in rows if r not in sample]
        sample.extend(rng.sample(remaining, n - len(sample)))

    return sample


@torch.no_grad()
def run_inference(rows: list[dict], checkpoint_path: str, device: torch.device) -> list[dict]:
    """Add scibert_b2_label, scibert_b2_prob, scibert_b3_label, scibert_b3_prob to each row."""
    tokenizer = load_tokenizer()

    # Build minimal rows with dummy labels for CriterionDataset
    padded = [{**r, 'b1_label': 0, 'b2_label': r['b2_label'] or 0,
               'b3_label': r['b3_label'] or 0,
               'b2_confidence': 0.0, 'b3_confidence': 0.0} for r in rows]

    dataset = CriterionDataset(padded, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CriterionClassifier().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    b2_labels, b2_probs = [], []
    b3_labels, b3_probs = [], []

    for batch in loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        logits = model(ids, mask)

        for head, labels_list, probs_list in (
            ('b2', b2_labels, b2_probs),
            ('b3', b3_labels, b3_probs),
        ):
            p = torch.softmax(logits[head], dim=-1)
            pred = p.argmax(dim=-1).cpu().tolist()
            prob_pos = p[:, 1].cpu().tolist()
            labels_list.extend(pred)
            probs_list.extend(prob_pos)

    for i, r in enumerate(rows):
        r['scibert_b2_label'] = b2_labels[i]
        r['scibert_b2_prob']  = round(b2_probs[i], 4)   # P(objective)
        r['scibert_b3_label'] = b3_labels[i]
        r['scibert_b3_prob']  = round(b3_probs[i], 4)   # P(observable)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',         default='data/processed/trials.duckdb')
    parser.add_argument('--checkpoint', default='nlp/checkpoints/best_model.pt')
    parser.add_argument('--n',          type=int, default=500)
    parser.add_argument('--seed',       type=int, default=89293)
    parser.add_argument('--out',        default='data/annotation/sample.csv')
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    print("Loading None rows...")
    rows = load_none_rows(args.db)
    print(f"  {len(rows):,} criteria with b2=None or b3=None")

    print(f"Sampling {args.n} (stratified by section)...")
    sample = stratified_sample(rows, args.n, args.seed)
    print(f"  Sampled {len(sample)}")

    print("Running SciBERT inference...")
    sample = run_inference(sample, args.checkpoint, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.DataFrame(sample)
    # Add empty columns for LLM and human labels — filled by subsequent scripts
    df['llm_b2_label']      = None   # 0/1
    df['llm_b2_confidence'] = None   # 0.0–1.0 (LLM self-reported)
    df['llm_b3_label']      = None
    df['llm_b3_confidence'] = None
    df['human_b2_label']    = None   # filled during review
    df['human_b3_label']    = None
    df['review_flag']       = None   # set by review script

    col_order = [
        'criterion_id', 'nct_id', 'section', 'text',
        'scibert_b2_label', 'scibert_b2_prob',
        'scibert_b3_label', 'scibert_b3_prob',
        'llm_b2_label', 'llm_b2_confidence',
        'llm_b3_label', 'llm_b3_confidence',
        'human_b2_label', 'human_b3_label',
        'review_flag',
    ]
    df[col_order].to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")

    # Summary
    print("\nSection breakdown:")
    print(df['section'].value_counts().to_string())
    print(f"\nSciBERT B2 predicted objective: {df['scibert_b2_label'].mean()*100:.1f}%")
    print(f"SciBERT B3 predicted observable: {df['scibert_b3_label'].mean()*100:.1f}%")


if __name__ == '__main__':
    main()
