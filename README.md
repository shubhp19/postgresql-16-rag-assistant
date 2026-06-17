# PostgreSQL 16 RAG Q&A Assistant

The system answers PostgreSQL 16 documentation questions using FAISS retrieval, local sentence-transformer embeddings, Groq LLM generation, source-grounded citations, off-topic refusal, follow-up handling, and retrieval evaluation.

Built with:

* Streamlit for the user interface
* FAISS for vector search
* sentence-transformers for local embeddings
* Groq API for answer generation
* Python evaluation scripts for Recall@k and answer similarity

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd pg_rag_app
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env
# Open .env and add:
# GROQ_API_KEY="your_groq_api_key_here"

# 3. Build the vector index
python build_index.py

# 4. Launch the app
streamlit run app.py
```

The index can also be built automatically on first app launch, or rebuilt from the Streamlit sidebar.

---

## Project Structure

```text
pg_rag_app/
├── app.py                         # Streamlit UI
├── build_index.py                 # One-shot index builder
├── requirements.txt               # Python dependencies
├── .env.example                   # Placeholder environment file only
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── ingest.py                  # Fetches and chunks PostgreSQL docs
│   ├── vectorstore.py             # Embeddings, FAISS index, dense retrieval
│   ├── hybrid_retriever.py        # BM25, hybrid retrieval, optional reranking
│   └── rag.py                     # Guardrails, prompt, LLM call, citation validation
├── eval/
│   ├── __init__.py
│   ├── qa_dataset.json            # 15 curated Q&A examples
│   ├── run_eval.py                # Recall@k and answer cosine evaluation
│   ├── chunk_ablation.py          # Chunk-size evaluation script
│   ├── retrieval_compare.py       # Dense/BM25/hybrid/reranker comparison
│   ├── chunk_ablation_results.csv
│   └── retrieval_compare_results.csv
└── data/                          # Created at runtime and ignored by Git
    ├── faiss.index
    └── chunks_meta.pkl

---

## Document Set

The app uses the PostgreSQL 16 documentation pages provided in the case study, including SQL commands and core concepts such as:

* SELECT, INSERT, UPDATE, DELETE, MERGE
* CREATE/ALTER/DROP TABLE
* CREATE/DROP INDEX
* EXPLAIN, ANALYZE, VACUUM
* BEGIN, COMMIT, ROLLBACK, SAVEPOINT
* GRANT, REVOKE, roles, databases
* COPY, TRUNCATE, SET, SHOW
* Indexes, partial indexes, constraints, MVCC, runtime configuration
* Materialized view refresh

---

## How It Works

### 1. Data Handling

The PostgreSQL 16 documentation pages are fetched directly from the official PostgreSQL website at index build time.

Each page is parsed with BeautifulSoup. Navigation, footer, and layout noise are removed before extracting the main documentation text. The cleaned text is split into overlapping word chunks using a 400-word chunk size and 80-word overlap.

The overlap helps preserve context when important explanations span chunk boundaries.

### 2. Embeddings and Storage

Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2`.

This model was selected because it is:

* lightweight,
* CPU-friendly,
* fast enough for a small case-study app,
* free to run locally,
* strong enough for semantic similarity retrieval.

The embeddings are L2-normalized so inner product search in FAISS behaves like cosine similarity. The FAISS index and chunk metadata are saved under `data/` and reused across app runs.

### 3. Retrieval

At query time, the user question is embedded with the same embedding model. FAISS retrieves the top-k most similar chunks.

The default retrieval value is `k=5`, and the user can adjust it in the Streamlit sidebar.

Each retrieved chunk includes:

* source ID,
* source URL,
* similarity score,
* chunk text.

These are shown in the app under the retrieved sources expander.

### 4. Prompt Design

The LLM receives:

* a system instruction,
* the last few chat turns,
* the retrieved PostgreSQL documentation chunks,
* the current user question.

The system prompt instructs the model to:

* answer only from the retrieved context,
* avoid unsupported claims,
* cite source IDs inline using `[Source: source_id]`,
* refuse questions outside the PostgreSQL documentation scope.

### 5. Chat History and Follow-up Handling

The app uses Streamlit session state to keep previous user and assistant messages during the chat.

For short follow-up questions, the app rewrites the query using the previous user question before retrieval. For example:

```text
Previous question: What is a materialized view?
Follow-up question: How do I refresh it?

### 6. Irrelevant Query Defense

The app includes a lightweight irrelevant-query defense before calling the LLM.

It checks:

1. whether the top retrieved chunk has a strong enough cosine similarity score,
2. whether the query contains PostgreSQL or SQL-related intent,
3. whether clearly non-document questions, jokes, personal questions, or unrelated requests should be rejected.

If the query appears unrelated to PostgreSQL 16 documentation, the app returns a polite refusal instead of calling the LLM.

This is intentionally simple for the case study. A production version could use a stronger intent classifier or a separate LLM-based query router.

### 7. Source Citation Validation

The LLM is instructed to cite retrieved sources inline. After generation, the app validates that cited source IDs actually belong to the retrieved chunks.

This helps prevent the model from inventing unsupported source IDs.

If an answer has no valid citation, the app adds a fallback source list based on the retrieved chunks. The Streamlit UI also shows retrieved source snippets separately, so the user can verify which documentation sections were used.

---

## Model and Retrieval Choices

### Selected Stack

| Layer                    | Selected Model / Tool                    | Why It Was Chosen                                                                                          |
| ------------------------ | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Embeddings               | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, CPU-friendly, fast, and does not require an embedding API key                                 |
| Dense Retrieval          | FAISS `IndexFlatIP`                      | Provides reliable semantic similarity search over normalized embeddings                                    |
| Lexical Retrieval        | BM25 via `rank-bm25`                     | Improves exact keyword matching for PostgreSQL terms such as `VACUUM`, `TRUNCATE`, `SAVEPOINT`, and `MVCC` |
| Final Retrieval Strategy | Hybrid FAISS + BM25                      | Best default tradeoff between retrieval quality and latency                                                |
| Optional Reranker        | `cross-encoder/ms-marco-MiniLM-L-6-v2`   | Improves top-k precision, but adds higher      latency                                                         
| LLM                      | `llama-3.1-8b-instant` via Groq          | Low-latency generation for documentation-grounded Q&A                                                      |
| UI                       | Streamlit                                | Simple interface for testing questions, viewing answers, and inspecting retrieved sources                  |

### Retrieval Strategy Evaluation

| Strategy                        | Recall@5 | Avg Latency |
| ------------------------------- | -------: | ----------: |
| Dense FAISS                     |    0.878 |    151.5 ms |
| BM25                            |    0.800 |      3.8 ms |
| Hybrid FAISS + BM25             |    0.933 |     17.5 ms |
| Dense + Cross-Encoder Reranker  |    0.911 |   2450.7 ms |
| Hybrid + Cross-Encoder Reranker |    0.944 |   1414.9 ms |

### Final Retrieval Decision

The final app uses **Hybrid FAISS + BM25** as the default retrieval strategy.

Dense retrieval gave a strong baseline with `0.878` Recall@5, but PostgreSQL documentation contains many exact technical terms where lexical matching is useful. Adding BM25 improved Recall@5 to `0.933` while keeping latency low.

The cross-encoder reranker achieved the highest Recall@5 at `0.944`, but it added significantly more latency. Because the recall improvement over hybrid retrieval was small compared with the latency cost, reranking was kept as an optional evaluation strategy instead of the default app strategy.

### Model Selection Rationale

`sentence-transformers/all-MiniLM-L6-v2` was selected for embeddings because the corpus is small, focused, and documentation-based. It is fast, local, CPU-friendly, and reproducible without relying on an external embedding API.

`llama-3.1-8b-instant` was selected for answer generation because this is a RAG application where the retrieved PostgreSQL documentation provides the factual grounding. The LLM mainly needs to summarize retrieved context, follow citation instructions, and answer clearly with low latency.


---
The final app uses hybrid retrieval as the default because PostgreSQL documentation contains many exact technical terms such as `VACUUM`, `TRUNCATE`, `SAVEPOINT`, `GIN`, and `MVCC`. Dense retrieval helps with semantic questions, while BM25 improves exact keyword matching. Cross-encoder reranking was included as an optional evaluation strategy, but not used as the default app strategy because it adds extra latency.

## Evaluation

The project includes a small QA dataset and evaluation script.

Run retrieval-only evaluation:

```bash
python -m eval.run_eval --no-llm --k 5
```

Run full evaluation with answer generation:

```bash
python -m eval.run_eval --k 5 --model llama-3.1-8b-instant
```

Evaluation results are saved to:

```text
eval/eval_results.csv
```

Then add this:

```md

### Evaluation Results

The RAG pipeline was evaluated on 15 curated PostgreSQL 16 Q&A examples. The evaluation measures both retrieval quality and generated answer quality.

| Metric                        | Result |
| ----------------------------- | -----: |
| Mean Recall@5                 |  0.933 |
| Mean Recall@10                |  0.967 |
| Mean Answer Cosine Similarity |  0.793 |

`Recall@5` measures whether the expected PostgreSQL documentation sources were retrieved in the top 5 chunks. `Recall@10` checks whether missed sources appear slightly lower in the ranking. `Mean Answer Cosine Similarity` compares the generated answer with the reference answer using embedding cosine similarity.

The final hybrid retrieval strategy improved Recall@5 compared with the earlier dense-only FAISS baseline. Dense retrieval achieved `0.878` Recall@5, while Hybrid FAISS + BM25 achieved `0.933` Recall@5. This shows that BM25 helped retrieve exact PostgreSQL terms such as `VACUUM`, `TRUNCATE`, `SAVEPOINT`, and `MVCC`, while dense retrieval handled semantic similarity.




### Chunk Size Ablation

| Chunk Size | Overlap | Number of Chunks | Recall@5 |
|---:|---:|---:|---:|
| 200 | 40 | 473 | 0.911 |
| 300 | 60 | 323 | 0.911 |
| 400 | 80 | 246 | 0.944 |
| 500 | 100 | 198 | 0.933 |
| 600 | 120 | 171 | 0.944 |

The final app uses 400-word chunks with 80-word overlap. Although 400 and 600 words both achieved the same Recall@5 of 0.944, 400-word chunks were selected because they provide more granular retrieval context while still maintaining strong recall.

### Metrics

**Recall@k**

Measures whether the expected source documents appear in the top-k retrieved chunks.

A score of 1.0 means all labelled relevant sources were retrieved for that question.

**Answer Cosine Similarity**

Measures semantic similarity between the generated answer and the reference answer using embeddings.

Values above 0.80 indicate strong semantic agreement. Values above 0.70 are acceptable for this minimal case-study evaluation.

### QA Dataset

`eval/qa_dataset.json` contains 15 curated Q&A examples covering:

* INSERT RETURNING
* DELETE vs TRUNCATE
* partial indexes
* VACUUM and MVCC
* transaction isolation
* EXPLAIN ANALYZE
* SAVEPOINT
* materialized views
* GRANT privileges
* B-tree vs GIN indexes
* MERGE
* constraints
* COPY
* transaction error behavior

Each question includes reference answers and expected source labels for Recall@k evaluation.

---

## Security

API keys are never committed to the repository.

`.env.example` contains only a placeholder:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

The real `.env` file is ignored by Git through `.gitignore`.

If a real API key was ever committed accidentally, it should be rotated before submitting or sharing the repository.

---
## Limitations

This project is intentionally small and explainable for a case study. Main limitations:

* **Fixed-size chunking:** the app uses word-based chunks, so some SQL examples or section-level context may still be split across chunks.
* **Simple hybrid weighting:** hybrid retrieval improves Recall@5, but the BM25/dense score weighting is still simple and could be tuned further.
* **Reranker not used by default:** cross-encoder reranking is implemented for evaluation, but it is not the default app strategy because it adds significant latency.
* **Lightweight irrelevant-query defense:** the current relevance guard is rule-based, not a trained production classifier.
* **Prompt-based grounding:** the model is instructed to answer only from retrieved PostgreSQL documentation, but LLMs can still make mistakes.
* **Basic citation validation:** cited source IDs are checked against retrieved chunks, but claim-level citation verification is not implemented.
* **Static index:** if PostgreSQL documentation changes, the index must be rebuilt.
* **English only:** the source documentation and embedding model are optimized for English.
* **Small evaluation set:** the evaluation set has 15 curated questions, which is useful for case-study comparison but not production-level validation.

## Future Improvements

Given more time, I would improve the system by adding:

* structure-aware chunking using PostgreSQL section headings,
* stronger query intent classification,
* claim-level citation verification,
* groundedness and hallucination evaluation,
* a UI toggle for dense, BM25, hybrid, and reranked retrieval,
* Docker packaging,
* deployment with persistent logging and monitoring.
- Add RAGAS-style evaluation for faithfulness, answer relevancy, context precision, and context recall.

---

## Dependencies

See `requirements.txt`.

Main packages:

* `streamlit`
* `groq`
* `sentence-transformers`
* `faiss-cpu`
* `requests`
* `beautifulsoup4`
* `numpy`
* `pandas`
* `scikit-learn`
* `python-dotenv`
