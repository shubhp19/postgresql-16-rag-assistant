"""
eval/run_eval.py
Measures retrieval quality (Recall@k) and answer similarity (embedding cosine)
against the QA dataset.

Usage:
    cd pg_rag_app
    python -m eval.run_eval [--k 5] [--model llama-3.1-8b-instant] [--no-llm]

--no-llm  : skip the LLM answer step (retrieval metrics only, no API key needed)
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore import load_index, index_exists, embed_texts, build_index
from src.ingest import load_all_chunks
from src.rag import answer as rag_answer

QA_PATH = Path(__file__).parent / "qa_dataset.json"


# --------------------------------------------------------------------------- #
#  Retrieval metric: Recall@k                                                  #
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved_source_ids: list[str], relevant_sources: list[str]) -> float:
    """
    Fraction of relevant sources that appear in the retrieved chunks.
    """
    if not relevant_sources:
        return 1.0
    retrieved_set = set(retrieved_source_ids)
    hits = sum(1 for s in relevant_sources if s in retrieved_set)
    return hits / len(relevant_sources)


# --------------------------------------------------------------------------- #
#  Answer quality metric: embedding cosine similarity                          #
# --------------------------------------------------------------------------- #
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def answer_similarity(predicted: str, reference: str) -> float:
    vecs = embed_texts([predicted, reference])
    return cosine_similarity(vecs[0], vecs[1])


# --------------------------------------------------------------------------- #
#  Main eval loop                                                               #
# --------------------------------------------------------------------------- #
def run(k: int = 5, model: str = "llama-3.1-8b-instant", no_llm: bool = False):
    with open(QA_PATH) as f:
        qa_set = json.load(f)

    if not index_exists():
        print("Index not found — building now (this takes a few minutes on first run)…")
        chunks = load_all_chunks(verbose=True)
        index, all_chunks = build_index(chunks)
    else:
        index, all_chunks = load_index()

    print(f"\nRunning eval on {len(qa_set)} questions  |  k={k}  |  LLM={'off' if no_llm else model}\n")
    print(f"{'ID':<6} {'Recall@k':>10} {'AnswerSim':>10}  Question")
    print("-" * 80)

    rows = []
    history: list[dict] = []   # shared history across eval (tests multi-turn robustness)

    for item in qa_set:
        qid       = item["id"]
        question  = item["question"]
        reference = item["reference_answer"]
        rel_srcs  = item["relevant_sources"]

        if no_llm:
            from src.vectorstore import retrieve
            top_chunks = retrieve(question, index, all_chunks, k=k)
            predicted  = ""
        else:
            result    = rag_answer(question, index, all_chunks, history=[], k=k, model=model)
            top_chunks= result["sources"]
            predicted = result["answer"]

        retrieved_src_ids = [c["source_id"] for c in top_chunks]
        rec_k   = recall_at_k(retrieved_src_ids, rel_srcs)
        ans_sim  = answer_similarity(predicted, reference) if predicted else float("nan")

        rows.append({
            "id":            qid,
            "question":      question,
            "recall_at_k":   rec_k,
            "answer_sim":    ans_sim,
            "predicted":     predicted,
            "reference":     reference,
            "retrieved_srcs":retrieved_src_ids,
            "relevant_srcs": rel_srcs,
        })

        sim_str = f"{ans_sim:.3f}" if not np.isnan(ans_sim) else "  N/A"
        print(f"{qid:<6} {rec_k:>10.3f} {sim_str:>10}  {question[:55]}…")

    # ----------- Summary ----------- #
    df = pd.DataFrame(rows)
    mean_recall = df["recall_at_k"].mean()
    mean_sim    = df["answer_sim"].dropna().mean()

    print("-" * 80)
    print(f"\n{'Mean Recall@' + str(k):<22}: {mean_recall:.3f}")
    if not no_llm:
        print(f"{'Mean Answer Cosine':<22}: {mean_sim:.3f}")

    # Save detailed results
    out_path = Path(__file__).parent / "eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved → {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the PG16 RAG pipeline")
    parser.add_argument("--k",      type=int, default=5,           help="Number of chunks to retrieve (default 5)")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Groq model for answer generation")
    parser.add_argument("--no-llm", action="store_true",           help="Skip LLM step; only measure retrieval")
    args = parser.parse_args()

    run(k=args.k, model=args.model, no_llm=args.no_llm)
