"""
build_index.py
One-shot script to fetch all PostgreSQL 16 docs, chunk them, embed,
and save the FAISS index to disk.

Run once before starting the app:
    python build_index.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.ingest import load_all_chunks
from src.vectorstore import build_index

if __name__ == "__main__":
    print("=== PostgreSQL 16 RAG — Index Builder ===\n")
    print("Step 1 / 2  Fetching and chunking documentation pages…")
    chunks = load_all_chunks(verbose=True)
    print(f"\nTotal chunks: {len(chunks)}\n")

    print("Step 2 / 2  Embedding and saving FAISS index…")
    build_index(chunks)
    print("\nDone. You can now run:  streamlit run app.py")
