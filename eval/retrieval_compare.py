"""
Compare retrieval strategies.

Usage:
    python -m eval.retrieval_compare
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore import load_index, index_exists, build_index
from src.ingest import load_all_chunks
from src.hybrid_retriever import retrieve_with_strategy
from eval.run_eval import recall_at_k


QA_PATH = Path(__file__).parent / "qa_dataset.json"
OUT_PATH = Path(__file__).parent / "retrieval_compare_results.csv"


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_set = json.load(f)

    if not index_exists():
        chunks = load_all_chunks(verbose=True)
        index, chunks = build_index(chunks)
    else:
        index, chunks = load_index()

    strategies = [
        "dense",
        "bm25",
        "hybrid",
        "rerank",
        "hybrid_rerank",
    ]

    rows = []

    for strategy in strategies:
        recalls = []
        latencies = []

        print(f"\nTesting strategy: {strategy}")

        for item in qa_set:
            start = time.perf_counter()

            top_chunks = retrieve_with_strategy(
                query=item["question"],
                index=index,
                chunks=chunks,
                k=5,
                strategy=strategy,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            retrieved_source_ids = [c["source_id"] for c in top_chunks]
            recall = recall_at_k(retrieved_source_ids, item["relevant_sources"])
            recalls.append(recall)

        row = {
            "strategy": strategy,
            "mean_recall_at_5": round(sum(recalls) / len(recalls), 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        }

        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print("\nRetrieval comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()