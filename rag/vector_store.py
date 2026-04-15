"""
ChromaDB vector store operations.

Collections:
  - trial_criteria: embedded criterion text with trial metadata
  - trial_summaries: embedded trial-level summaries for coarse retrieval
"""


def get_client(persist_dir: str = "./chroma_db"):
    pass


def upsert_criteria(client, criteria: list[dict]) -> None:
    pass


def query_criteria(client, query_embedding: list[float], n_results: int = 20) -> list[dict]:
    pass
