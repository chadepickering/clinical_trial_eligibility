"""
Training loop for the multi-task SciBERT classifier.

Loads weakly labeled criteria from DuckDB, splits train/val 80/20,
trains with AdamW + linear warmup, saves best checkpoint by val macro-F1.

W&B logging is a planned addition (Step 4 sub-step) — not yet implemented.

Usage:
    python -m nlp.trainer                          # default config
    python -m nlp.trainer --db data/processed/trials.duckdb --epochs 5
"""

import argparse
import os
import random

import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from ingestion.database import get_connection
from nlp.multitask_classifier import (
    CriterionClassifier,
    CriterionDataset,
    compute_loss,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    'db':           'data/processed/trials.duckdb',
    'epochs':       8,
    'batch_size':   32,
    'lr':           2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,       # fraction of total steps used for warmup
    'val_split':    0.2,
    'seed':         48697,
    'checkpoint_dir': 'nlp/checkpoints',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_criteria(db_path: str) -> list[dict]:
    """Load all criteria rows from DuckDB as plain dicts."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT text, b1_label, b2_label, b3_label, b2_confidence, b3_confidence
        FROM criteria
        WHERE text IS NOT NULL AND length(text) > 0
    """).fetchall()
    conn.close()

    cols = ['text', 'b1_label', 'b2_label', 'b3_label', 'b2_confidence', 'b3_confidence']
    return [dict(zip(cols, r)) for r in rows]


def train_val_split(rows: list[dict], val_fraction: float, seed: int):
    """Reproducible random split into train and val index lists."""
    indices = list(range(len(rows)))
    random.seed(seed)
    random.shuffle(indices)
    split = int(len(indices) * (1 - val_fraction))
    return indices[:split], indices[split:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """
    Run inference on a DataLoader and return per-head F1 scores.

    Only examples with non-zero confidence contribute to F1 for that head —
    zero-confidence rows had None labels and should not be evaluated.
    """
    model.eval()

    preds = {'b1': [], 'b2': [], 'b3': []}
    golds = {'b1': [], 'b2': [], 'b3': []}
    confs = {'b1': [], 'b2': [], 'b3': []}

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)

        for head in ('b1', 'b2', 'b3'):
            pred = logits[head].argmax(dim=-1).cpu().tolist()
            gold = batch[f'{head}_label'].tolist()
            conf = batch[f'{head}_conf'].tolist()
            preds[head].extend(pred)
            golds[head].extend(gold)
            confs[head].extend(conf)

    results = {}
    for head in ('b1', 'b2', 'b3'):
        # Filter to rows that had a real label (conf > 0)
        labeled = [(g, p) for g, p, c in zip(golds[head], preds[head], confs[head]) if c > 0.0]
        if len(labeled) < 2:
            results[f'f1_{head}'] = 0.0
            results[f'f1_{head}_pos'] = 0.0
            results[f'f1_{head}_neg'] = 0.0
            continue
        g_vals, p_vals = zip(*labeled)
        results[f'f1_{head}'] = f1_score(g_vals, p_vals, average='binary', zero_division=0)
        per_class = f1_score(g_vals, p_vals, average=None, labels=[0, 1], zero_division=0)
        results[f'f1_{head}_neg'] = float(per_class[0])  # label=0 (exclusion/subjective/unobservable)
        results[f'f1_{head}_pos'] = float(per_class[1])  # label=1 (inclusion/objective/observable)

    macro_keys = ('f1_b1', 'f1_b2', 'f1_b3')
    results['f1_macro'] = sum(results[k] for k in macro_keys) / len(macro_keys)
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # --- Data ---
    print(f"Loading criteria from {config['db']}...", flush=True)
    rows = load_criteria(config['db'])
    print(f"  {len(rows)} criteria loaded", flush=True)

    tokenizer = load_tokenizer()
    dataset = CriterionDataset(rows, tokenizer)

    train_idx, val_idx = train_val_split(rows, config['val_split'], config['seed'])
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}", flush=True)

    # --- Model ---
    print("Loading SciBERT...", flush=True)
    model = CriterionClassifier().to(device)

    # --- Optimizer + scheduler ---
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Checkpoint dir ---
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_f1 = 0.0
    best_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')

    # --- Loop ---
    print(f"\nTraining for {config['epochs']} epochs...", flush=True)
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['input_ids'], batch['attention_mask'])

            loss = compute_loss(logits, batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)}  loss={epoch_loss/step:.4f}", flush=True)

        avg_loss = epoch_loss / len(train_loader)
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{config['epochs']}  loss={avg_loss:.4f}  macro={metrics['f1_macro']:.3f}\n"
            f"  B1  macro={metrics['f1_b1']:.3f}  excl={metrics['f1_b1_neg']:.3f}  incl={metrics['f1_b1_pos']:.3f}\n"
            f"  B2  macro={metrics['f1_b2']:.3f}  subj={metrics['f1_b2_neg']:.3f}  obj={metrics['f1_b2_pos']:.3f}\n"
            f"  B3  macro={metrics['f1_b3']:.3f}  unobs={metrics['f1_b3_neg']:.3f}  obs={metrics['f1_b3_pos']:.3f}",
            flush=True
        )

        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config,
            }, best_path)
            print(f"  Checkpoint saved (macro F1={best_f1:.3f})", flush=True)

    print(f"\nTraining complete. Best macro F1: {best_f1:.3f}")
    print(f"Checkpoint: {best_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-task SciBERT criterion classifier")
    parser.add_argument('--db',           default=DEFAULTS['db'])
    parser.add_argument('--epochs',       type=int,   default=DEFAULTS['epochs'])
    parser.add_argument('--batch-size',   type=int,   default=DEFAULTS['batch_size'])
    parser.add_argument('--lr',           type=float, default=DEFAULTS['lr'])
    parser.add_argument('--checkpoint-dir',           default=DEFAULTS['checkpoint_dir'])
    parser.add_argument('--seed',         type=int,   default=DEFAULTS['seed'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = {**DEFAULTS, **{k.replace('-', '_'): v for k, v in vars(args).items()}}
    train(config)
