"""
RAGAS evaluation suite for the RAG pipeline.

Metrics:
  - Faithfulness: generated answer grounded in retrieved context
  - Answer relevancy: answer addresses the query
  - Context precision / recall: retrieval quality against reference set
"""


def evaluate_pipeline(eval_dataset: list[dict]) -> dict:
    pass
