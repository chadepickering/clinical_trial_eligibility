"""
Trial-level embedding pipeline for semantic retrieval.

Embeds a composite document per trial (brief_title + brief_summary +
eligibility_text) using sentence-transformers/all-MiniLM-L6-v2.

Strategy:
    - Single-pass truncation to model max sequence length (256 tokens).
      Title + summary together almost always fit; eligibility text is appended
      and the tail is truncated. Acceptable for portfolio-scale retrieval;
      mean-pooled chunk approach can replace this if quality suffers.
    - normalize_embeddings=True enables cosine similarity via dot product in
      ChromaDB (faster than explicit cosine).
    - Batch size 64 balances memory and throughput on CPU (~15 min for 15k).

Usage (called from embed.py):
    from rag.embedder import build_corpus, embed_corpus
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_corpus(rows: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    Build composite embedding texts from trial rows.

    Args:
        rows: list of dicts with keys nct_id, brief_title, brief_summary,
              eligibility_text, conditions, phases, status.

    Returns:
        texts     — one composite string per trial
        metadatas — one metadata dict per trial (for ChromaDB)
        ids       — nct_id per trial (ChromaDB document IDs)
    """
    texts, metadatas, ids = [], [], []

    for row in rows:
        parts = [row.get('brief_title') or '']
        if row.get('brief_summary'):
            parts.append(row['brief_summary'])
        if row.get('eligibility_text'):
            parts.append(row['eligibility_text'])
        composite = ' '.join(p.strip() for p in parts if p.strip())

        texts.append(composite)
        metadatas.append({
            'nct_id':     row['nct_id'],
            'conditions': str(row.get('conditions') or ''),
            'phases':     str(row.get('phases') or ''),
            'status':     str(row.get('status') or ''),
        })
        ids.append(row['nct_id'])

    return texts, metadatas, ids


def embed_corpus(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Encode a list of texts into 384-dim normalised embeddings.

    Args:
        texts:         list of composite trial strings
        batch_size:    sentences per encoding batch
        show_progress: display tqdm progress bar

    Returns:
        list of embedding vectors (list[float], length 384)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.tolist()
