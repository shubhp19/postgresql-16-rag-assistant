"""
hybrid_retriever.py

Adds:
1. BM25 lexical retrieval
2. Hybrid dense + BM25 retrieval
3. Optional cross-encoder reranking
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.vectorstore import retrieve


_BM25_CACHE = {}
_RERANKER = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _get_bm25(chunks: list[dict]) -> BM25Okapi:
    cache_key = id(chunks)

    if cache_key not in _BM25_CACHE:
        corpus = [
            _tokenize(chunk["source_id"] + " " + chunk["text"])
            for chunk in chunks
        ]
        _BM25_CACHE[cache_key] = BM25Okapi(corpus)

    return _BM25_CACHE[cache_key]


def bm25_retrieve(query: str, chunks: list[dict], k: int = 10) -> list[dict]:
    bm25 = _get_bm25(chunks)
    scores = bm25.get_scores(_tokenize(query))

    top_indices = np.argsort(scores)[::-1][:k]
    max_score = max(float(scores[i]) for i in top_indices) if len(top_indices) else 1.0
    max_score = max(max_score, 1e-9)

    results = []

    for rank, idx in enumerate(top_indices):
        chunk = dict(chunks[idx])
        chunk["bm25_score"] = float(scores[idx]) / max_score
        chunk["score"] = chunk["bm25_score"]
        chunk["rank"] = rank + 1
        results.append(chunk)

    return results


def _normalize_dense_results(results: list[dict]) -> dict:
    if not results:
        return {}

    scores = [r.get("score", 0.0) for r in results]
    min_s = min(scores)
    max_s = max(scores)
    denom = max(max_s - min_s, 1e-9)

    normalized = {}

    for r in results:
        chunk_id = r["chunk_id"]
        normalized[chunk_id] = {
            "chunk": r,
            "dense_score": (r.get("score", 0.0) - min_s) / denom,
        }

    return normalized


def hybrid_retrieve(
    query: str,
    index,
    chunks: list[dict],
    k: int = 5,
    candidate_k: int = 20,
    dense_weight: float = 0.6,
) -> list[dict]:
    """
    Combines dense FAISS retrieval with BM25 lexical retrieval.

    dense_weight=0.6 means:
        final_score = 0.6 * dense_score + 0.4 * bm25_score
    """
    dense_results = retrieve(query, index, chunks, k=candidate_k)
    bm25_results = bm25_retrieve(query, chunks, k=candidate_k)

    merged = {}

    for item in dense_results:
        merged[item["chunk_id"]] = {
            "chunk": item,
            "dense_score": item.get("score", 0.0),
            "bm25_score": 0.0,
        }

    normalized_dense = _normalize_dense_results(dense_results)

    for chunk_id, payload in normalized_dense.items():
        merged[chunk_id]["dense_score"] = payload["dense_score"]

    for item in bm25_results:
        chunk_id = item["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = {
                "chunk": item,
                "dense_score": 0.0,
                "bm25_score": item.get("bm25_score", 0.0),
            }
        else:
            merged[chunk_id]["bm25_score"] = item.get("bm25_score", 0.0)

    ranked = []

    for payload in merged.values():
        hybrid_score = (
            dense_weight * payload["dense_score"]
            + (1 - dense_weight) * payload["bm25_score"]
        )

        chunk = dict(payload["chunk"])
        chunk["dense_score"] = float(payload["dense_score"])
        chunk["bm25_score"] = float(payload["bm25_score"])
        chunk["score"] = float(hybrid_score)
        ranked.append(chunk)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(ranked[:k]):
        item["rank"] = rank + 1

    return ranked[:k]


def _get_reranker() -> CrossEncoder:
    global _RERANKER

    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _RERANKER


def rerank(
    query: str,
    candidates: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    Reranks candidate chunks using a free local cross-encoder.
    More accurate than vector similarity alone, but slower.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    reranked = []

    for chunk, score in zip(candidates, scores):
        item = dict(chunk)
        item["rerank_score"] = float(score)
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    for rank, item in enumerate(reranked[:k]):
        item["rank"] = rank + 1

    return reranked[:k]


def retrieve_with_strategy(
    query: str,
    index,
    chunks: list[dict],
    k: int = 5,
    strategy: str = "dense",
) -> list[dict]:
    """
    strategy options:
      - dense
      - bm25
      - hybrid
      - rerank
      - hybrid_rerank
    """
    if strategy == "dense":
        return retrieve(query, index, chunks, k=k)

    if strategy == "bm25":
        return bm25_retrieve(query, chunks, k=k)

    if strategy == "hybrid":
        return hybrid_retrieve(query, index, chunks, k=k, candidate_k=20)

    if strategy == "rerank":
        candidates = retrieve(query, index, chunks, k=20)
        return rerank(query, candidates, k=k)

    if strategy == "hybrid_rerank":
        candidates = hybrid_retrieve(query, index, chunks, k=20, candidate_k=30)
        return rerank(query, candidates, k=k)

    raise ValueError(f"Unknown retrieval strategy: {strategy}")