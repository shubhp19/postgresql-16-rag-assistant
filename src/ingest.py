"""
ingest.py
Fetches PostgreSQL 16 documentation pages, cleans them,
chunks text into overlapping windows, and returns a list
of chunk dicts ready for embedding.
"""

import re
import time
import requests
from bs4 import BeautifulSoup

DOCS = [
    ("sql-select", "https://www.postgresql.org/docs/16/sql-select.html"),
    ("sql-insert", "https://www.postgresql.org/docs/16/sql-insert.html"),
    ("sql-update", "https://www.postgresql.org/docs/16/sql-update.html"),
    ("sql-delete", "https://www.postgresql.org/docs/16/sql-delete.html"),
    ("sql-merge", "https://www.postgresql.org/docs/16/sql-merge.html"),

    ("sql-createtable", "https://www.postgresql.org/docs/16/sql-createtable.html"),
    ("sql-altertable", "https://www.postgresql.org/docs/16/sql-altertable.html"),
    ("sql-droptable", "https://www.postgresql.org/docs/16/sql-droptable.html"),
    ("sql-createindex", "https://www.postgresql.org/docs/16/sql-createindex.html"),
    ("sql-dropindex", "https://www.postgresql.org/docs/16/sql-dropindex.html"),

    ("sql-explain", "https://www.postgresql.org/docs/16/sql-explain.html"),
    ("sql-analyze", "https://www.postgresql.org/docs/16/sql-analyze.html"),
    ("sql-vacuum", "https://www.postgresql.org/docs/16/sql-vacuum.html"),

    ("sql-begin", "https://www.postgresql.org/docs/16/sql-begin.html"),
    ("sql-commit", "https://www.postgresql.org/docs/16/sql-commit.html"),
    ("sql-rollback", "https://www.postgresql.org/docs/16/sql-rollback.html"),
    ("sql-savepoint", "https://www.postgresql.org/docs/16/sql-savepoint.html"),
    ("sql-set-transaction", "https://www.postgresql.org/docs/16/sql-set-transaction.html"),

    ("sql-createview", "https://www.postgresql.org/docs/16/sql-createview.html"),
    ("sql-grant", "https://www.postgresql.org/docs/16/sql-grant.html"),
    ("sql-revoke", "https://www.postgresql.org/docs/16/sql-revoke.html"),
    ("sql-copy", "https://www.postgresql.org/docs/16/sql-copy.html"),
    ("sql-truncate", "https://www.postgresql.org/docs/16/sql-truncate.html"),
    ("sql-set", "https://www.postgresql.org/docs/16/sql-set.html"),
    ("sql-show", "https://www.postgresql.org/docs/16/sql-show.html"),

    ("sql-createrole", "https://www.postgresql.org/docs/16/sql-createrole.html"),
    ("sql-alterrole", "https://www.postgresql.org/docs/16/sql-alterrole.html"),
    ("sql-createdatabase", "https://www.postgresql.org/docs/16/sql-createdatabase.html"),
    ("sql-dropdatabase", "https://www.postgresql.org/docs/16/sql-dropdatabase.html"),

    ("indexes", "https://www.postgresql.org/docs/16/indexes.html"),
    ("indexes-partial", "https://www.postgresql.org/docs/16/indexes-partial.html"),
    ("ddl-constraints", "https://www.postgresql.org/docs/16/ddl-constraints.html"),
    ("mvcc", "https://www.postgresql.org/docs/16/mvcc.html"),
    ("runtime-config", "https://www.postgresql.org/docs/16/runtime-config.html"),
    ("sql-refreshmaterializedview", "https://www.postgresql.org/docs/16/sql-refreshmaterializedview.html"),
]

CHUNK_SIZE   = 400   # words per chunk
CHUNK_OVERLAP= 80    # word overlap between consecutive chunks


def _fetch_text(url: str) -> str:
    """Download a page and return its main body as plain text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # PostgreSQL docs keep content inside <div class="sect1"> / <div class="refentry">
    # Fallback to <body> if neither found.
    main = (
        soup.find("div", {"class": "refentry"})
        or soup.find("div", {"id": "docContent"})
        or soup.find("body")
    )
    if main is None:
        return ""

    # Drop nav, toc, footer noise
    for tag in main.find_all(["nav", "footer", "script", "style", "table"]):
        tag.decompose()

    text = main.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk(text: str, source_id: str, url: str):
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + CHUNK_SIZE]
        if len(window) < 20:          # skip tiny tail fragments
            continue
        chunks.append({
            "chunk_id": f"{source_id}__c{i}",
            "source_id": source_id,
            "url": url,
            "text": " ".join(window),
        })
    return chunks


def load_all_chunks(verbose: bool = True) -> list[dict]:
    """Fetch all docs and return flat list of chunk dicts."""
    all_chunks = []
    for source_id, url in DOCS:
        if verbose:
            print(f"  Fetching {source_id} ...", end=" ", flush=True)
        try:
            text = _fetch_text(url)
            chunks = _chunk(text, source_id, url)
            all_chunks.extend(chunks)
            if verbose:
                print(f"{len(chunks)} chunks")
        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")
        time.sleep(0.3)   # be polite to the server
    return all_chunks
