"""
app.py – Streamlit UI for the PostgreSQL 16 RAG Q&A app
Run with:  streamlit run app.py
"""

import os
import sys
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Make sure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from src.ingest import load_all_chunks
from src.vectorstore import build_index, load_index, index_exists
from src.rag import answer
def clean_citations(text: str) -> str:
    """
    Force citations to use only source_id.
    Example:
    [Source: sql-insert | URL: https://...] -> [Source: sql-insert]
    """
    return re.sub(
        r"\[Source:\s*([A-Za-z0-9_-]+).*?\]",
        r"[Source: \1]",
        text,
    )
# --------------------------------------------------------------------------- #
#  Page config                                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="PostgreSQL 16 Doc Assistant",
    page_icon="🐘",
    layout="wide",
)

# --------------------------------------------------------------------------- #
#  Sidebar – settings                                                           #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🐘 PG16 Doc Assistant")
    st.markdown("Ask anything about PostgreSQL 16 — SELECT, indexes, MVCC, roles, and more.")
    st.divider()
        

#    api_key = st.text_input("Groq API Key",
#         type="password",
#         value=os.environ.get("GROQ_API_KEY", ""),
#         help="Required to generate answers. Your key is never stored.",
#     )
#     if api_key:
#         os.environ["GROQ_API_KEY"] = api_key
        # Prefer server-side key from environment or Streamlit secrets.
        # Prefer server-side key from .env, environment variable, or Streamlit secrets.
    server_api_key = os.environ.get("GROQ_API_KEY", "")

    try:
        server_api_key = st.secrets.get("GROQ_API_KEY", server_api_key)
    except Exception:
        pass

    # Do not pre-fill the key in the UI.
    user_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value="",
        help="Optional if a server-side key is configured. Your key is only used for this session.",
    )

    if user_api_key:
        os.environ["GROQ_API_KEY"] = user_api_key
    elif server_api_key:
        os.environ["GROQ_API_KEY"] = server_api_key

    if server_api_key:
        st.caption("✅ API key loaded from server environment.")
    else:
        st.caption("Enter a Groq API key to generate answers.")

    model_choice = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama3-70b-8192"], index=0)
    top_k = st.slider("Top-K chunks to retrieve", min_value=2, max_value=10, value=5)

    st.divider()
    rebuild = st.button("🔄 Rebuild index from docs", use_container_width=True)
    if rebuild:
        with st.spinner("Fetching and indexing PostgreSQL docs…"):
            chunks = load_all_chunks(verbose=False)
            st.session_state["index"], st.session_state["chunks"] = build_index(chunks)
        st.success(f"Index built — {len(st.session_state['chunks'])} chunks indexed.")

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["messages"] = []
        st.rerun()

    st.caption("Embeddings: all-MiniLM-L6-v2 (local)  |  Retrieval: Hybrid FAISS cosine + BM25")

# --------------------------------------------------------------------------- #
#  Index bootstrapping                                                          #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading index…")
def get_index_and_chunks():
    if index_exists():
        return load_index()
    # First run: fetch and index everything
    chunks = load_all_chunks(verbose=False)
    return build_index(chunks)


if "index" not in st.session_state or "chunks" not in st.session_state:
    idx, cks = get_index_and_chunks()
    st.session_state["index"]  = idx
    st.session_state["chunks"] = cks

if "history" not in st.session_state:
    st.session_state["history"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []   # display messages (role, content, sources)

# --------------------------------------------------------------------------- #
#  Main chat UI                                                                 #
# --------------------------------------------------------------------------- #
st.title("🐘 PostgreSQL 16 Documentation Assistant")
st.caption(
    f"Answers grounded in the official PostgreSQL 16 docs. "
    f"Using **{model_choice}** · top-{top_k} retrieval."
)

# Render previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(clean_citations(msg["content"]))
        if msg.get("sources"):
            with st.expander(f"📄 Retrieved sources ({len(msg['sources'])} chunks)", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['source_id']}]({s['url']})** &nbsp; score: `{s['score']:.3f}`\n\n"
                        f"> {s['text'][:300]}…"
                    )
# Chat input
query = st.chat_input("Ask a question about PostgreSQL 16…")

if query and query.strip():
    query = query.strip()

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Build history directly from displayed chat messages
    history_for_rag = [
        {"role": m["role"], "content": clean_citations(m["content"])}
        for m in st.session_state["messages"][-6:]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]

    # Default query sent to RAG
    query_for_rag = query

    # Follow-up detection
    q_lower = query.lower().strip()

    direct_question_starts = (
        "what is",
        "what are",
        "define",
        "explain ",
        "tell me about",
    )

    followup_signals = (
        " it",
        " one",
        " this",
        " that",
        " they",
        " them",
        "how do i",
        "how can i",
        "when would i",
        "why would i",
        "can you give",
        "give an example",
        "show example",
        "show me",
        "what about",
        "how about",
    )

    short_followup = len(query.split()) <= 12
    looks_like_direct_question = q_lower.startswith(direct_question_starts)
    looks_like_followup = any(signal in f" {q_lower}" for signal in followup_signals)

    previous_user_question = None

    for m in reversed(st.session_state["messages"]):
        if m.get("role") == "user" and m.get("content"):
            previous_user_question = m["content"]
            break

    if (
        short_followup
        and looks_like_followup
        and not looks_like_direct_question
        and previous_user_question
    ):
        query_for_rag = (
            f"Previous question: {previous_user_question}\n"
            f"Follow-up question: {query}"
        )

    # Show user message immediately
    st.session_state["messages"].append({
        "role": "user",
        "content": query,
        "sources": [],
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = answer(
                query=query_for_rag,
                index=st.session_state["index"],
                all_chunks=st.session_state["chunks"],
                history=history_for_rag,
                k=top_k,
                model=model_choice,
            )

        result["answer"] = clean_citations(result["answer"])
        st.markdown(result["answer"])

        if not result["relevant"]:
            st.info("⚠️ This question appears to be outside the PostgreSQL documentation scope.")

        if result["sources"]:
            with st.expander(f"📄 Retrieved sources ({len(result['sources'])} chunks)", expanded=False):
                for s in result["sources"]:
                    st.markdown(
                        f"**[{s['source_id']}]({s['url']})** &nbsp; score: `{s['score']:.3f}`\n\n"
                        f"> {s['text'][:300]}…"
                    )

    # Save display history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

    # Keep RAG history synced
    st.session_state["history"] = [
        {"role": m["role"], "content": clean_citations(m["content"])}
        for m in st.session_state["messages"][-6:]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]