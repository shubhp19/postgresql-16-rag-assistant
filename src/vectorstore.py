"""
vectorstore.py
Builds (or loads from disk) a FAISS index over chunk embeddings.
Uses sentence-transformers so no API key is needed for embeddings.
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "data/faiss.index"
META_PATH   = "data/chunks_meta.pkl"
MODEL_NAME  = "all-MiniLM-L6-v2"   # fast, 384-dim, runs CPU-only

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")


def build_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """Embed all chunks and build a flat IP (cosine) FAISS index."""
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks with {MODEL_NAME} ...")
    vecs = embed_texts(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine when vecs are L2-normalised
    index.add(vecs)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved index ({index.ntotal} vectors) → {INDEX_PATH}")
    return index, chunks


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load a previously built index from disk."""
    index  = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def retrieve(query: str, index: faiss.Index, chunks: list[dict], k: int = 5) -> list[dict]:
    """Return the top-k most relevant chunks for a query."""
    q_vec = embed_texts([query])                   # shape (1, dim)
    scores, indices = index.search(q_vec, k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        chunk["rank"]  = rank + 1
        results.append(chunk)
    return results
