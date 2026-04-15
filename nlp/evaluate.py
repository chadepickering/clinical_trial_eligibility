"""
Evaluation for multi-task classifier and NER extractor.

Computes per-task F1, precision, recall against a held-out labeled set.
Outputs a classification report for each of B1, B2, B3.
"""


def evaluate_classifier(model, dataloader) -> dict:
    pass


def evaluate_ner(model, examples: list[dict]) -> dict:
    pass
