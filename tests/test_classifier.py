"""
Unit tests for nlp/multitask_classifier.py and nlp/trainer.py.

All tests are fast — no SciBERT download, no DuckDB.
The tokenizer is mocked to isolate logic under test.
"""

import pytest
import torch
from unittest.mock import MagicMock

from nlp.multitask_classifier import (
    B1_CONFIDENCE,
    MAX_LENGTH,
    CriterionDataset,
    compute_loss,
)
from nlp.trainer import train_val_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tokenizer():
    """Minimal mock tokenizer that returns fixed-shape tensors."""
    def _call(text, max_length, padding, truncation, return_tensors):
        return {
            'input_ids':      torch.zeros(1, max_length, dtype=torch.long),
            'attention_mask': torch.ones(1, max_length, dtype=torch.long),
        }
    return MagicMock(side_effect=_call)


def make_row(
    text='Age >= 18 years',
    b1_label=1, b2_label=1, b3_label=1,
    b2_confidence=0.8, b3_confidence=0.75,
):
    return {
        'text': text,
        'b1_label': b1_label,
        'b2_label': b2_label,
        'b3_label': b3_label,
        'b2_confidence': b2_confidence,
        'b3_confidence': b3_confidence,
    }


def make_batch(rows, tokenizer):
    dataset = CriterionDataset(rows, tokenizer)
    items = [dataset[i] for i in range(len(rows))]
    return {k: torch.stack([item[k] for item in items]) for k in items[0]}


def make_logits(batch_size=2, num_classes=2):
    return {
        'b1': torch.randn(batch_size, num_classes),
        'b2': torch.randn(batch_size, num_classes),
        'b3': torch.randn(batch_size, num_classes),
    }


# ---------------------------------------------------------------------------
# CriterionDataset — tensor shapes
# ---------------------------------------------------------------------------

def test_dataset_returns_all_expected_keys():
    dataset = CriterionDataset([make_row()], make_tokenizer())
    expected = {'input_ids', 'attention_mask', 'b1_label', 'b2_label', 'b3_label',
                'b1_conf', 'b2_conf', 'b3_conf'}
    assert expected == set(dataset[0].keys())


def test_dataset_input_ids_shape():
    dataset = CriterionDataset([make_row()], make_tokenizer())
    assert dataset[0]['input_ids'].shape == torch.Size([MAX_LENGTH])


def test_dataset_attention_mask_shape():
    dataset = CriterionDataset([make_row()], make_tokenizer())
    assert dataset[0]['attention_mask'].shape == torch.Size([MAX_LENGTH])


def test_dataset_label_tensors_are_long():
    dataset = CriterionDataset([make_row()], make_tokenizer())
    item = dataset[0]
    for key in ('b1_label', 'b2_label', 'b3_label'):
        assert item[key].dtype == torch.long, f"{key} should be long"


def test_dataset_conf_tensors_are_float():
    dataset = CriterionDataset([make_row()], make_tokenizer())
    item = dataset[0]
    for key in ('b1_conf', 'b2_conf', 'b3_conf'):
        assert item[key].dtype == torch.float, f"{key} should be float"


def test_dataset_len():
    dataset = CriterionDataset([make_row(), make_row(), make_row()], make_tokenizer())
    assert len(dataset) == 3


# ---------------------------------------------------------------------------
# CriterionDataset — None label handling
# ---------------------------------------------------------------------------

def test_none_b1_label_gives_zero_conf():
    dataset = CriterionDataset([make_row(b1_label=None)], make_tokenizer())
    assert dataset[0]['b1_conf'].item() == 0.0


def test_none_b2_label_gives_zero_conf():
    dataset = CriterionDataset([make_row(b2_label=None, b2_confidence=0.0)], make_tokenizer())
    assert dataset[0]['b2_conf'].item() == 0.0


def test_none_b3_label_gives_zero_conf():
    dataset = CriterionDataset([make_row(b3_label=None, b3_confidence=0.0)], make_tokenizer())
    assert dataset[0]['b3_conf'].item() == 0.0


def test_none_b1_label_converted_to_zero_int():
    dataset = CriterionDataset([make_row(b1_label=None)], make_tokenizer())
    assert dataset[0]['b1_label'].item() == 0


def test_none_b2_label_converted_to_zero_int():
    dataset = CriterionDataset([make_row(b2_label=None, b2_confidence=0.0)], make_tokenizer())
    assert dataset[0]['b2_label'].item() == 0


def test_none_b3_label_converted_to_zero_int():
    dataset = CriterionDataset([make_row(b3_label=None, b3_confidence=0.0)], make_tokenizer())
    assert dataset[0]['b3_label'].item() == 0


def test_b1_labeled_row_uses_fixed_confidence():
    dataset = CriterionDataset([make_row(b1_label=1)], make_tokenizer())
    assert dataset[0]['b1_conf'].item() == pytest.approx(B1_CONFIDENCE)


def test_b1_none_row_uses_zero_confidence():
    dataset = CriterionDataset([make_row(b1_label=None)], make_tokenizer())
    assert dataset[0]['b1_conf'].item() == pytest.approx(0.0)


def test_b2_confidence_preserved_when_labeled():
    dataset = CriterionDataset([make_row(b2_label=1, b2_confidence=0.67)], make_tokenizer())
    assert dataset[0]['b2_conf'].item() == pytest.approx(0.67)


def test_b3_confidence_preserved_when_labeled():
    dataset = CriterionDataset([make_row(b3_label=0, b3_confidence=0.55)], make_tokenizer())
    assert dataset[0]['b3_conf'].item() == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# compute_loss — return type and properties
# ---------------------------------------------------------------------------

def test_compute_loss_returns_scalar():
    batch = make_batch([make_row(), make_row()], make_tokenizer())
    loss = compute_loss(make_logits(batch_size=2), batch)
    assert loss.shape == torch.Size([])


def test_compute_loss_is_finite():
    batch = make_batch([make_row(), make_row()], make_tokenizer())
    loss = compute_loss(make_logits(batch_size=2), batch)
    assert torch.isfinite(loss)


def test_compute_loss_is_nonnegative():
    batch = make_batch([make_row(), make_row()], make_tokenizer())
    loss = compute_loss(make_logits(batch_size=2), batch)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# compute_loss — confidence weighting
# ---------------------------------------------------------------------------

def test_zero_conf_rows_produce_zero_loss():
    """A batch where all confidences are 0.0 should produce loss=0."""
    row = make_row(b1_label=None, b2_label=None, b3_label=None,
                   b2_confidence=0.0, b3_confidence=0.0)
    batch = make_batch([row, row], make_tokenizer())
    loss = compute_loss(make_logits(batch_size=2), batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_all_zero_conf_does_not_produce_nan():
    """The clamp(min=1e-8) NaN guard must hold when all confs are 0."""
    row = make_row(b1_label=None, b2_label=None, b3_label=None,
                   b2_confidence=0.0, b3_confidence=0.0)
    batch = make_batch([row, row], make_tokenizer())
    loss = compute_loss(make_logits(batch_size=2), batch)
    assert not torch.isnan(loss)


def test_custom_task_weights_change_loss():
    """Zeroing a task weight should produce a different (lower) loss."""
    batch = make_batch([make_row(), make_row()], make_tokenizer())
    torch.manual_seed(1)
    logits = make_logits(batch_size=2)
    loss_default = compute_loss(logits, batch)
    loss_no_b3   = compute_loss(logits, batch, task_weights={'b1': 0.5, 'b2': 0.5, 'b3': 0.0})
    assert loss_default.item() != loss_no_b3.item()
    assert torch.isfinite(loss_no_b3)


def test_loss_supports_backprop():
    """Loss must have grad_fn so optimizer.step() can run."""
    batch = make_batch([make_row(), make_row()], make_tokenizer())
    logits = {k: v.requires_grad_(True) for k, v in make_logits(batch_size=2).items()}
    loss = compute_loss(logits, batch)
    assert loss.requires_grad


# ---------------------------------------------------------------------------
# train_val_split
# ---------------------------------------------------------------------------

def test_train_val_split_sizes():
    train_idx, val_idx = train_val_split(list(range(100)), val_fraction=0.2, seed=42)
    assert len(train_idx) == 80
    assert len(val_idx) == 20


def test_train_val_split_no_overlap():
    train_idx, val_idx = train_val_split(list(range(200)), val_fraction=0.2, seed=42)
    assert len(set(train_idx) & set(val_idx)) == 0


def test_train_val_split_covers_all_indices():
    train_idx, val_idx = train_val_split(list(range(200)), val_fraction=0.2, seed=42)
    assert sorted(train_idx + val_idx) == list(range(200))


def test_train_val_split_reproducible():
    rows = list(range(500))
    t1, v1 = train_val_split(rows, val_fraction=0.2, seed=99)
    t2, v2 = train_val_split(rows, val_fraction=0.2, seed=99)
    assert t1 == t2
    assert v1 == v2


def test_train_val_split_different_seeds_differ():
    rows = list(range(500))
    t1, _ = train_val_split(rows, val_fraction=0.2, seed=1)
    t2, _ = train_val_split(rows, val_fraction=0.2, seed=2)
    assert t1 != t2
