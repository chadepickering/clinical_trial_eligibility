"""
Retrieval and cross-encoder reranking pipeline.

Steps:
  1. Embed patient query with all-MiniLM-L6-v2
  2. Retrieve top-k candidates from ChromaDB
  3. Rerank with cross-encoder (ms-marco-MiniLM-L-6-v2)
  4. Return top-n reranked trials/criteria
"""


def retrieve_and_rerank(query: str, n_retrieve: int = 50, n_return: int = 10) -> list[dict]:
    pass
