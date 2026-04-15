"""
Sentence-level embeddings using sentence-transformers.

Model: all-MiniLM-L6-v2 (fast, 384-dim, strong semantic retrieval)
Encodes criterion text and patient profile fields for ChromaDB indexing.
"""


def embed_texts(texts: list[str]) -> list[list[float]]:
    pass
