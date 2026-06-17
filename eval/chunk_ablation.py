"""
Evaluate chunk sizes for retrieval quality.

Usage:
    python -m eval.chunk_ablation
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.ingest as ingest
from src.vectorstore import embed_texts, retrieve
from eval.run_eval import recall_at_k


QA_PATH = Path(__file__).parent / "qa_dataset.json"
OUT_PATH = Path(__file__).parent / "chunk_ablation_results.csv"


def build_temp_index(chunks: list[dict]) -> faiss.Index:
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.array(vecs, dtype="float32"))
    return index


def evaluate_chunk_size(chunk_size: int, overlap_ratio: float = 0.2, k: int = 5) -> dict:
    ingest.CHUNK_SIZE = chunk_size
    ingest.CHUNK_OVERLAP = int(chunk_size * overlap_ratio)

    print(f"\nTesting chunk_size={chunk_size}, overlap={ingest.CHUNK_OVERLAP}")

    chunks = ingest.load_all_chunks(verbose=False)
    index = build_temp_index(chunks)

    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_set = json.load(f)

    recalls = []

    for item in qa_set:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        top_chunks = retrieve(question, index, chunks, k=k)
        retrieved_source_ids = [c["source_id"] for c in top_chunks]

        recalls.append(recall_at_k(retrieved_source_ids, relevant_sources))

    mean_recall = sum(recalls) / len(recalls)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": ingest.CHUNK_OVERLAP,
        "num_chunks": len(chunks),
        f"mean_recall_at_{k}": round(mean_recall, 3),
    }


def main():
    chunk_sizes = [200, 300, 400, 500, 600]
    rows = []

    for size in chunk_sizes:
        row = evaluate_chunk_size(size, overlap_ratio=0.2, k=5)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print("\nChunk ablation results:")
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()