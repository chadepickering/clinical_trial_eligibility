"""
Multi-task SciBERT classifier for eligibility criterion classification.

Three binary classification heads sharing a single SciBERT encoder:
    B1 — Inclusion (1) vs Exclusion (0)
    B2 — Objective (1) vs Subjective (0)
    B3 — Observable (1) vs Unobservable (0)

Loss is confidence-weighted: each criterion's contribution to a head's loss
is scaled by the weak labeler's confidence score for that label. Criteria with
confidence=0.0 (i.e. None labels) contribute zero loss, degenerating to
standard masking. Criteria with partial confidence are weighted proportionally,
so uncertain labels inform training without being treated as ground truth.

B1 has no confidence score in the weak labeler (rule-based, ~0.90 reliable),
so B1 loss uses a fixed weight of 0.90 for labeled rows and 0.0 for None rows.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL = 'allenai/scibert_scivocab_uncased'
B1_CONFIDENCE = 0.90          # fixed confidence for rule-derived B1 labels
MAX_LENGTH = 128               # max tokens per criterion — criteria are short


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CriterionDataset(Dataset):
    """
    Loads criterion rows from a list of dicts (as returned from DuckDB).

    Each row must have: text, b1_label, b2_label, b3_label,
                        b2_confidence, b3_confidence.

    Labels of None are converted to 0 with confidence 0.0 so the loss
    weight zeroes them out — they participate in forward passes (for
    efficiency) but contribute nothing to gradient updates.
    """

    def __init__(self, rows: list[dict], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        encoding = self.tokenizer(
            row['text'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # B1: label + fixed confidence (0.0 if label is None)
        b1_label = row['b1_label'] if row['b1_label'] is not None else 0
        b1_conf = B1_CONFIDENCE if row['b1_label'] is not None else 0.0

        # B2/B3: label + labeler confidence (already 0.0 when label is None)
        b2_label = row['b2_label'] if row['b2_label'] is not None else 0
        b2_conf = row['b2_confidence'] if row['b2_label'] is not None else 0.0

        b3_label = row['b3_label'] if row['b3_label'] is not None else 0
        b3_conf = row['b3_confidence'] if row['b3_label'] is not None else 0.0

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'b1_label':       torch.tensor(b1_label, dtype=torch.long),
            'b2_label':       torch.tensor(b2_label, dtype=torch.long),
            'b3_label':       torch.tensor(b3_label, dtype=torch.long),
            'b1_conf':        torch.tensor(b1_conf,  dtype=torch.float),
            'b2_conf':        torch.tensor(b2_conf,  dtype=torch.float),
            'b3_conf':        torch.tensor(b3_conf,  dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CriterionClassifier(nn.Module):
    """
    Shared SciBERT encoder with three independent classification heads.

    Forward pass returns a dict of logits, one per head. Loss is computed
    externally via compute_loss() to keep the model reusable for inference
    (where no labels are available).
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size  # 768 for SciBERT

        self.dropout = nn.Dropout(dropout)

        # Independent heads — no shared parameters after the CLS token
        self.head_b1 = nn.Linear(hidden_size, 2)
        self.head_b2 = nn.Linear(hidden_size, 2)
        self.head_b3 = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])  # [batch, 768]

        return {
            'b1': self.head_b1(cls),   # [batch, 2]
            'b2': self.head_b2(cls),   # [batch, 2]
            'b3': self.head_b3(cls),   # [batch, 2]
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    task_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Confidence-weighted multi-task cross-entropy loss.

    For each head, per-example loss is scaled by the weak label confidence
    for that head. A confidence of 0.0 zeroes out that example's contribution
    (equivalent to masking). A confidence of 1.0 treats it as a gold label.

    task_weights controls relative importance across heads:
        b1: 0.30 — reliable but structurally simple (section context)
        b2: 0.30 — moderate signal, ~55% labeled
        b3: 0.40 — most consequential errors downstream (Bayesian model)

    Returns a scalar tensor for backprop.
    """
    if task_weights is None:
        task_weights = {'b1': 0.30, 'b2': 0.30, 'b3': 0.40}

    # Class weights to counteract minority-class suppression.
    # B1 is balanced (~1:1) so no correction needed.
    # B2 subjective:objective ≈ 1:3  → upweight subjective (label=0) by 3×
    # B3 unobservable:observable ≈ 1:8 → upweight unobservable (label=0) by 8×
    # Weight vector order: [label=0, label=1]
    device = logits['b1'].device
    class_weights = {
        'b1': None,
        'b2': torch.tensor([3.0, 1.0], dtype=torch.float, device=device),
        'b3': torch.tensor([8.0, 1.0], dtype=torch.float, device=device),
    }

    total_loss = torch.tensor(0.0, device=device)

    for head in ('b1', 'b2', 'b3'):
        ce = nn.CrossEntropyLoss(reduction='none', weight=class_weights[head])
        per_example = ce(logits[head], batch[f'{head}_label'])   # [batch]
        conf = batch[f'{head}_conf'].to(device)                  # [batch]
        # Guard against all-zero confidence in a batch (would produce NaN mean)
        denom = conf.sum().clamp(min=1e-8)
        weighted = (per_example * conf).sum() / denom
        total_loss = total_loss + task_weights[head] * weighted

    return total_loss


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer(base_model: str = BASE_MODEL) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(base_model)
