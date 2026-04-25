"""
Trial-level embedding pipeline for semantic retrieval.

Embeds a composite document per trial (brief_title + brief_summary +
eligibility_text) using sentence-transformers/all-MiniLM-L6-v2.

Strategy — mean pooling of overlapping chunks:
    The model truncates at 256 tokens. A composite trial document (title +
    summary + eligibility text) typically runs 500–700 tokens. Single-pass
    truncation discards the eligibility criteria, which are the primary
    patient-matching signal. Instead, the document is split into overlapping
    256-token chunks, each chunk is encoded independently, and the resulting
    unit vectors are mean-pooled then L2-normalised. This ensures all content
    contributes to the final embedding.

    Chunk size:    256 tokens (model max)
    Overlap:        32 tokens (preserves cross-boundary context)
    Pool:          mean of chunk embeddings
    Normalisation: explicit L2 after pooling (mean of unit vectors ≠ unit vector)

    Throughput: ~2–3× single-pass (most documents produce 2–3 chunks).
    Runtime:    ~2–3 minutes for 15k trials on CPU.

Usage (called from embed.py):
    from rag.embedder import build_corpus, embed_corpus
"""

import math

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 256   # tokens
CHUNK_OVERLAP = 32  # tokens

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _chunk_text(text: str, tokenizer) -> list[str]:
    """
    Split text into overlapping token-boundary chunks and decode back to strings.

    Uses the model's own tokenizer so chunk boundaries are aligned with what
    the encoder actually sees. Special tokens ([CLS], [SEP]) are excluded from
    the token budget — the tokenizer adds them at encode time.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= CHUNK_SIZE:
        return [text]

    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + CHUNK_SIZE]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if start + CHUNK_SIZE >= len(token_ids):
            break

    return chunks


def _embed_and_pool(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Encode chunks, mean-pool, then L2-normalise.

    Each chunk is encoded with normalize_embeddings=True (unit vectors).
    Their mean is not itself a unit vector, so explicit L2 normalisation
    is applied after pooling.
    """
    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    pooled = chunk_embeddings.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled


def build_corpus(rows: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    Build composite embedding texts from trial rows.

    Composite order: title → brief_summary → eligibility_text.
    brief_summary names specific drugs, biomarkers, and trial design — the most
    distinctive content. Eligibility text follows; with mean pooling, it is no
    longer truncated so all inclusion/exclusion criteria contribute signal.

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


def embed_one(text: str) -> list[float]:
    """Embed a single string. Convenience wrapper around embed_corpus."""
    return embed_corpus([text], show_progress=False)[0]


def embed_corpus(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Encode a list of composite trial texts into 384-dim normalised embeddings
    using mean pooling over overlapping chunks.

    Documents that fit within CHUNK_SIZE tokens are encoded in a single pass
    (same result as single-pass truncation). Longer documents are chunked,
    each chunk encoded independently, and the embeddings mean-pooled and
    L2-normalised.

    Short documents are batched together via model.encode for efficiency.
    Long documents (requiring chunking) are processed one at a time — they
    are a minority of the corpus.

    Args:
        texts:         list of composite trial strings
        batch_size:    documents per encoding batch (short-doc fast path)
        show_progress: display tqdm progress bar

    Returns:
        list of embedding vectors (list[float], length 384)
    """
    model = _get_model()
    tokenizer = model.tokenizer

    # Partition: short docs go through the fast batch path; long docs are chunked
    short_indices, short_texts = [], []
    long_indices, long_texts = [], []

    for i, text in enumerate(texts):
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens <= CHUNK_SIZE:
            short_indices.append(i)
            short_texts.append(text)
        else:
            long_indices.append(i)
            long_texts.append(text)

    embeddings = [None] * len(texts)

    # Fast path — batch encode short documents
    if short_texts:
        short_embs = model.encode(
            short_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress and not long_texts,
            convert_to_numpy=True,
        )
        for idx, emb in zip(short_indices, short_embs):
            embeddings[idx] = emb.tolist()

    # Chunked path — mean pool long documents
    if long_texts:
        if show_progress:
            from tqdm import tqdm
            long_iter = tqdm(
                zip(long_indices, long_texts),
                total=len(long_texts),
                desc="Chunked docs",
            )
        else:
            long_iter = zip(long_indices, long_texts)

        for idx, text in long_iter:
            chunks = _chunk_text(text, tokenizer)
            pooled = _embed_and_pool(chunks, model)
            embeddings[idx] = pooled.tolist()

    return embeddings
