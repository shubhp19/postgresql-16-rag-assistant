"""
rag.py
Handles the full RAG pipeline:
  - top-k retrieval
  - follow-up aware retrieval
  - irrelevant query guard
  - prompt assembly
  - Groq LLM call
  - citation cleanup / validation
  - chat history management
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv
from groq import Groq

#from src.vectorstore import retrieve
from src.hybrid_retriever import retrieve_with_strategy


load_dotenv()


# --------------------------------------------------------------------------- #
#  PostgreSQL / SQL topic keywords                                             #
# --------------------------------------------------------------------------- #

TOPIC_KEYWORDS = {
    "select", "insert", "update", "delete", "merge", "table", "index",
    "vacuum", "analyze", "explain", "transaction", "commit", "rollback",
    "roll back", "savepoint", "release savepoint", "view", "materialized",
    "grant", "revoke", "role", "database", "schema", "constraint",
    "primary key", "foreign key", "unique", "check", "sequence", "trigger",
    "function", "mvcc", "concurrency", "lock", "isolation", "copy",
    "truncate", "partition", "query", "sql", "postgres", "postgresql",
    "join", "where", "group", "having", "order", "limit", "offset", "cte",
    "window", "aggregate", "subquery", "cast", "type", "null", "serial",
    "identity", "default", "column", "row", "record", "cluster",
    "tablespace", "reindex", "bloat", "toast", "heap", "btree", "hash",
    "gin", "gist", "partial index", "runtime", "config", "setting", "show",
    "set", "performance", "difference", "compare", "versus", "locking",
    "scan", "disk", "space",
}

OFF_TOPIC_PATTERNS = {
    "joke", "poem", "song", "weather", "stock", "sports", "cricket",
    "football", "movie", "recipe", "resume", "cover letter", "travel",
    "flight", "news", "politics", "astrology", "girlfriend", "dating",
    "capital of", "who are you", "your name", "salary", "bank", "credit card",
}

MIN_STRONG_SCORE = 0.30
MIN_WEAK_SCORE = 0.10


# --------------------------------------------------------------------------- #
#  System prompt                                                               #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a helpful PostgreSQL 16 documentation assistant.

Answer questions using ONLY the provided PostgreSQL 16 documentation context.

Rules:
- Be direct and concise.
- Prefer 3 to 6 bullets for comparison questions.
- Do not say "According to the provided context".
- Do not mention previous questions unless the current question is clearly a follow-up.
- Always include source citations inline using this exact format: [Source: source_id]
- Do not include URLs inside citations.
- Do not add behavior differences unless they are clearly present in the retrieved context.
- Avoid transaction claims unless the retrieved context explicitly supports them.
- Do not invent source IDs.

If the context does not contain enough information, say so clearly.
Do not answer unrelated questions."""


# --------------------------------------------------------------------------- #
#  Guardrail helpers                                                           #
# --------------------------------------------------------------------------- #

def _has_postgres_intent(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in TOPIC_KEYWORDS)


def _looks_off_topic(query: str) -> bool:
    q = query.lower()
    return any(pattern in q for pattern in OFF_TOPIC_PATTERNS)


def _is_followup_like(query: str) -> bool:
    """
    Detect contextual follow-ups like:
      - When would I use one?
      - How?
      - Why?
      - Show syntax.
      - Give example.
    """
    q = query.lower().strip()

    followup_terms = {
        "it", "this", "that", "one", "they", "them",
        "how", "why", "when", "where", "which",
        "example", "examples", "syntax", "use", "used",
        "faster", "slower", "difference", "compare",
        "what about", "how about", "show me", "explain more",
        "roll back", "rollback", "release",
    }

    if len(q.split()) <= 12:
        return True

    return any(term in q for term in followup_terms)


def _history_has_postgres_context(history: list[dict]) -> bool:
    """
    Checks whether recent chat history was PostgreSQL-related.
    This lets short follow-ups pass when the previous topic was valid.
    """
    if not history:
        return False

    recent_text = " ".join(
        turn.get("content", "")
        for turn in history[-6:]
    )

    return _has_postgres_intent(recent_text)


def _combined_query_for_retrieval(query: str, history: list[dict]) -> str:
    """
    Builds a better retrieval query.

    For follow-ups, use the latest conversation turns + current query.
    This helps:
      "What is a partial index?"
      "When would I use one?"
    retrieve partial-index context.
    """
    if not history:
        return query

    recent_turns = history[-4:]
    recent_context = "\n".join(
        turn.get("content", "")
        for turn in recent_turns
        if turn.get("content")
    )

    return f"{recent_context}\n{query}"


def _is_relevant(
    query: str,
    retrieval_query: str,
    top_chunks: list[dict],
    history: list[dict],
) -> bool:
    """
    Simple reliable relevance guard:
      1. Reject obvious off-topic requests.
      2. Allow short follow-ups when chat history exists.
      3. Allow direct PostgreSQL/SQL questions.
      4. Allow strong retrieval matches.
    """
    if not top_chunks:
        return False

    # Still block obvious unrelated requests like jokes/weather/sports.
    if _looks_off_topic(query):
        return False
    
    if history and _is_followup_like(query):
        return True

    top_score = top_chunks[0].get("score", 0.0)

    # Direct PostgreSQL/SQL question.
    if _has_postgres_intent(query):
        return True

    # Short follow-up after a valid previous answer.
    # Fixes: "When would I use one?", "How?", "Why?", "Show example."
    if history and _is_followup_like(query):
        return True

    # Strong semantic match.
    if top_score >= MIN_STRONG_SCORE:
        return True

    # Combined retrieval query has PostgreSQL context.
    if history and _has_postgres_intent(retrieval_query) and top_score >= MIN_WEAK_SCORE:
        return True

    return False

# --------------------------------------------------------------------------- #
#  Groq client                                                                 #
# --------------------------------------------------------------------------- #

def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env or enter it in the Streamlit sidebar."
        )

    return Groq(api_key=api_key)


# --------------------------------------------------------------------------- #
#  Prompt assembly                                                             #
# --------------------------------------------------------------------------- #

def build_messages(
    query: str,
    chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """
    Build messages for the LLM.

    Important:
    We do NOT pass old chat history to the LLM because it can contaminate
    the answer with previous topics like VACUUM, indexes, etc.

    Follow-up context should already be handled by query rewriting in app.py
    and retrieval_query in answer().
    """
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"Source ID: {chunk['source_id']}\n"
            f"Text: {chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    user_content = (
        f"Use only the PostgreSQL 16 documentation context below.\n\n"
        f"Context:\n\n"
        f"{context_block}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer the question directly. Do not refer to previous questions unless the current question explicitly asks for a follow-up."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
#  Citation cleanup / validation                                               #
# --------------------------------------------------------------------------- #

def validate_and_fix_citations(
    answer_text: str,
    retrieved_chunks: list[dict],
) -> tuple[str, bool]:
    """
    Normalizes and validates citations.

    Converts:
      [Source: sql-insert | URL: https://...]
    into:
      [Source: sql-insert]

    Removes fake source IDs.
    """
    allowed_source_ids = {chunk["source_id"] for chunk in retrieved_chunks}

    def normalize_source(match: re.Match) -> str:
        raw = match.group(1).strip()
        source_id = raw.split("|")[0].strip()

        if source_id in allowed_source_ids:
            return f"[Source: {source_id}]"

        return ""

    cleaned_answer = re.sub(
        r"\[Source:\s*([^\]]+)\]",
        normalize_source,
        answer_text,
    ).strip()

    cited_source_ids = set(re.findall(r"\[Source:\s*([^\]]+)\]", cleaned_answer))
    valid_citations = cited_source_ids.intersection(allowed_source_ids)

    if not valid_citations and retrieved_chunks:
        fallback_sources = ", ".join(
            f"[Source: {chunk['source_id']}]" for chunk in retrieved_chunks[:3]
        )
        cleaned_answer = cleaned_answer + f"\n\nSources used: {fallback_sources}"
        return cleaned_answer, False

    return cleaned_answer, True


# --------------------------------------------------------------------------- #
#  Main RAG entry point                                                        #
# --------------------------------------------------------------------------- #

def answer(
    query: str,
    index,
    all_chunks: list[dict],
    history: list[dict],
    k: int = 5,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Full RAG cycle:
      1. Build follow-up aware retrieval query.
      2. Retrieve top-k chunks.
      3. Check relevance.
      4. Call Groq LLM.
      5. Validate citations.
      6. Return answer, sources, and updated history.
    """
    retrieval_query = _combined_query_for_retrieval(query, history)
    top_chunks = retrieve_with_strategy(
    query=retrieval_query,
    index=index,
    chunks=all_chunks,
    k=k,
    strategy="hybrid",
)

    if not _is_relevant(query, retrieval_query, top_chunks, history):
        off_topic_reply = (
            "I can only answer questions about PostgreSQL 16. "
            "Your question doesn't seem to be related to PostgreSQL. "
            "Please ask about SQL commands, indexes, transactions, roles, "
            "MVCC, constraints, or other PostgreSQL features."
        )
        

        return {
            "answer": off_topic_reply,
            "sources": [],
            "relevant": False,
            "citation_valid": True,
            "history": history,
        }

    messages = build_messages(query, top_chunks, history)
    client = _get_groq_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        reply = response.choices[0].message.content or ""

    except Exception as exc:
        return {
            "answer": (
                "I retrieved relevant PostgreSQL documentation, but the LLM call failed. "
                f"Error: {str(exc)}"
            ),
            "sources": top_chunks,
            "relevant": True,
            "citation_valid": False,
            "history": history,
        }

    reply, citation_valid = validate_and_fix_citations(reply, top_chunks)

    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": reply},
    ]

    return {
        "answer": reply,
        "sources": top_chunks,
        "relevant": True,
        "citation_valid": citation_valid,
        "history": updated_history,
    }