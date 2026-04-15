"""
Multi-task SciBERT classifier with three classification heads.

Shared SciBERT encoder with task-specific linear heads:
  - Head B1: Inclusion vs Exclusion (binary)
  - Head B2: Objective vs Subjective (binary)
  - Head B3: Observable vs Unobservable (binary)

Loss: weighted sum of three binary cross-entropy losses.
"""
import torch
import torch.nn as nn


class MultiTaskCriterionClassifier(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        super().__init__()
        # TODO: load SciBERT encoder and attach three classification heads

    def forward(self, input_ids, attention_mask):
        # TODO: shared encoding → three head logits
        pass
