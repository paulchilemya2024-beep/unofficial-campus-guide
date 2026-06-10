"""
query.py — Retrieval and Grounded Generation
Unofficial Campus Survival Guide RAG System

Milestone 5: Semantic retrieval + grounded LLM generation with source attribution.
"""

import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from embed import get_collection

TOP_K = 4  # number of chunks to retrieve per query

# Grounding system prompt — instructs the LLM to use ONLY retrieved context
SYSTEM_PROMPT = """You are the Unofficial Campus Survival Guide — a knowledgeable peer advisor \
who gives practical, grounded advice to college students.

CRITICAL RULES:
1. Answer ONLY using the information in the provided documents below.
2. Do NOT use your general training knowledge to fill in gaps.
3. If the documents don't contain enough information to answer the question, say:
   "I don't have enough information on that in my sources. Try asking your school's \
Dean of Students office or r/college."
4. Be specific and actionable. Students need real advice, not vague platitudes.
5. Cite which document(s) you drew from at the end of your answer using the format:
   Sources: [filename1, filename2]

You are helping a real student navigate a real problem. Be direct and helpful."""


# ── Load models once (module-level for performance) ──────────────────────────

_embedding_model = None
_groq_client = None
_collection = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file or environment."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_chroma_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks.

    Returns list of dicts with keys: text, source, chunk_index, distance.
    Distance is cosine distance (lower = more similar).
    """
    model = _get_embedding_model()
    query_embedding = model.encode(query).tolist()

    collection = _get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
        })

    return chunks


# ── Generation ────────────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the grounded user prompt — question + retrieved context."""
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Document {i}: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""Here are the relevant documents from the Unofficial Campus Survival Guide:

{context}

---

Student question: {question}

Answer based only on the documents above. Be specific and practical. \
End with: Sources: [list the document filenames you used]."""


def ask(question: str, k: int = TOP_K, verbose: bool = False) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

    Returns:
        {
          "answer": str,        # LLM response
          "sources": list[str], # source document filenames
          "chunks": list[dict], # retrieved chunks (for inspection)
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(question, k=k)

    if verbose:
        print(f"\n[query] Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"  - {c['source']} (distance: {c['distance']})")
            print(f"    {c['text'][:120]}...")

    # Step 2: Build grounded prompt
    prompt = build_prompt(question, chunks)

    # Step 3: Generate
    client = _get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,   # lower temp = more faithful to retrieved context
        max_tokens=600,
    )

    answer = response.choices[0].message.content.strip()

    # Extract source filenames from retrieved chunks (guaranteed, not left to LLM)
    sources = list(dict.fromkeys(c["source"] for c in chunks))  # deduplicated, ordered

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


# ── Quick CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_questions = [
        "What should I do if my financial aid was reduced or cut?",
        "How do I get off a course waitlist?",
        "What mental health resources does my campus have?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("="*60)
        result = ask(q, verbose=True)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
