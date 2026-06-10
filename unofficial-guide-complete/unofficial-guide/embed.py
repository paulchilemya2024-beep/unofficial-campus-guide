"""
embed.py — Embedding and Vector Store Setup
Unofficial Campus Survival Guide RAG System

Milestone 4: Embed all chunks with all-MiniLM-L6-v2 and store in ChromaDB.
Run this once to build the vector store. Re-run only if documents change.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from ingest import run_pipeline

COLLECTION_NAME = "campus_survival_guide"
CHROMA_PATH = "./chroma_db"


def build_vector_store(chunks: list[dict], reset: bool = False) -> chromadb.Collection:
    """
    Embed all chunks and store in a persistent ChromaDB collection.

    Args:
        chunks: Output from ingest.run_pipeline()
        reset: If True, delete existing collection and rebuild from scratch.

    Returns:
        The ChromaDB collection, ready to query.
    """
    # Load embedding model (runs locally, no API key needed)
    print("[embed] Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Optionally reset
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[embed] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"[embed] Collection already has {existing_count} chunks. Skipping embed.")
        print("[embed] Pass reset=True to rebuild from scratch.")
        return collection

    # Extract texts and metadata
    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    # Embed all chunks
    print(f"[embed] Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    

    # Store in ChromaDB in batches (ChromaDB has a default batch limit)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_end = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:batch_end],
            embeddings=embeddings[i:batch_end],
            ids=ids[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"[embed] Stored chunks {i}–{batch_end}")

    print(f"[embed] Done. {collection.count()} chunks in vector store.")
    return collection


def get_collection() -> chromadb.Collection:
    """Load the existing ChromaDB collection (for use in query.py)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    chunks = run_pipeline()
    collection = build_vector_store(chunks, reset=True)
    print(f"\n[embed] Vector store ready at '{CHROMA_PATH}'")
    print(f"[embed] Collection '{COLLECTION_NAME}' contains {collection.count()} chunks.")
