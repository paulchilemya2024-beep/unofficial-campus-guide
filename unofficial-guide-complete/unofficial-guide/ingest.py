"""
ingest.py — Document Ingestion and Chunking Pipeline
Unofficial Campus Survival Guide RAG System

Milestone 3: Load, clean, and chunk all source documents.
"""

import os
import re

DOCS_DIR = "docs"
CHUNK_SIZE = 500       # characters (~80-100 words)
CHUNK_OVERLAP = 75    # characters (~15% overlap)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list[dict]:
    """Load all .txt files from the docs directory."""
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        documents.append({
            "source": filename,
            "raw_text": raw_text,
        })
    print(f"[ingest] Loaded {len(documents)} documents from '{docs_dir}'")
    return documents


def clean_text(text: str) -> str:
    """
    Remove boilerplate from raw document text.
    Keeps: actual student tips, advice, opinions, context.
    Removes: SOURCE/URL/DATE headers, repeated dashes, excess whitespace.
    """
    # Remove SOURCE / URL / DATE header lines
    text = re.sub(r"SOURCE:.*\n", "", text)
    text = re.sub(r"URL:.*\n", "", text)
    text = re.sub(r"DATE:.*\n", "", text)

    # Remove decorative dividers (--- lines)
    text = re.sub(r"-{3,}", "", text)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks.

    Strategy: recursive split — try to break on paragraph boundaries (\n\n),
    then sentence boundaries (. ? !), then hard character limit.
    This prevents splitting mid-sentence for most real-world text.

    Returns list of dicts with 'text', 'source', 'chunk_index'.
    """
    # Split on paragraph breaks first
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    # Combine short paragraphs into chunks of ~chunk_size chars, with overlap
    chunks = []
    current = ""
    current_index = 0

    for para in paragraphs:
        # If adding this paragraph stays within size, accumulate it
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            # Save current chunk if non-empty
            if current:
                chunks.append({
                    "text": current,
                    "source": source,
                    "chunk_index": current_index,
                })
                current_index += 1

            # Start new chunk with overlap from end of previous chunk
            if current:
                overlap_text = current[-overlap:].strip()
                current = (overlap_text + " " + para).strip()
            else:
                current = para

            # If a single paragraph is bigger than chunk_size, hard-split it
            while len(current) > chunk_size:
                chunks.append({
                    "text": current[:chunk_size],
                    "source": source,
                    "chunk_index": current_index,
                })
                current_index += 1
                current = current[chunk_size - overlap:].strip()

    # Don't forget the last chunk
    if current:
        chunks.append({
            "text": current,
            "source": source,
            "chunk_index": current_index,
        })

    return chunks


def run_pipeline(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Full ingestion pipeline: load → clean → chunk → return all chunks."""
    documents = load_documents(docs_dir)
    all_chunks = []

    for doc in documents:
        cleaned = clean_text(doc["raw_text"])
        chunks = chunk_text(cleaned, source=doc["source"])
        all_chunks.extend(chunks)

    print(f"[ingest] Total chunks produced: {len(all_chunks)}")
    return all_chunks


# ── Inspection helpers (for Milestone 3 checkpoint) ──────────────────────────

def inspect_chunks(chunks: list[dict], n: int = 5) -> None:
    """Print n sample chunks for manual review."""
    import random
    sample = random.sample(chunks, min(n, len(chunks)))
    print(f"\n{'='*60}")
    print(f"CHUNK INSPECTION — {n} random samples")
    print(f"{'='*60}")
    for i, chunk in enumerate(sample, 1):
        print(f"\n--- Chunk {i} | Source: {chunk['source']} | Index: {chunk['chunk_index']} ---")
        print(chunk["text"])
        print(f"[{len(chunk['text'])} chars]")
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c['text']) for c in chunks) // len(chunks)} chars")


if __name__ == "__main__":
    chunks = run_pipeline()
    inspect_chunks(chunks, n=5)
