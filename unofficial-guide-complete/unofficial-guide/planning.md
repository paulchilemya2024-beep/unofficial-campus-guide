# planning.md — The Unofficial Campus Survival Guide

## Domain
**General college survival knowledge**: registration tips, financial aid navigation, mental health resources, and campus support services. This knowledge is hard to find through official channels because universities publish polished, institutional language that obscures the real experience — what actually happens when you miss a FAFSA deadline, how to get a fee waived, which campus counseling services have the shortest wait times, how to appeal a grade or a financial aid decision. Students learn this from each other through Reddit, Discord, and word of mouth — not the registrar's website.

## Documents
At least 10 sources covering different subtopics within campus survival:

1. r/college — Reddit threads on registration, add/drop deadlines, waitlists
2. r/financialaid — Reddit threads on FAFSA, aid appeals, scholarship tips
3. r/mentalhealth (college-focused threads) — campus counseling experiences
4. r/college — threads on academic probation, GPA recovery
5. r/college — threads on talking to professors, office hours, extensions
6. r/StudentLoans — tips on loan repayment, deferment, IDR plans
7. College Confidential forums — housing lottery, roommate conflict advice
8. r/college — first-generation student tips, imposter syndrome
9. Reddit AMA threads with financial aid officers
10. r/college — part-time job balancing, work-study experiences
11. Student wellness blog posts (publicly available) — stress management
12. r/college — how to use campus food pantries, emergency funds

**Variety check**: Sources span registration, money, mental health, academics, and social survival — different subtopics that together answer a wide range of questions.

## Chunking Strategy

**Document type**: Short-to-medium Reddit posts and comments (50–400 words each). Some blog posts run longer (500–1000 words). Key facts are usually concentrated in 1–3 sentences. Opinions and tips are self-contained.

**Chosen strategy**:
- Chunk size: **500 characters** (~80–100 words)
- Overlap: **75 characters** (~15% of chunk size)
- Split method: recursive character splitting, preferring paragraph and sentence boundaries over hard character cuts

**Why these numbers fit the documents**:
- Reddit comments are short and opinionated — a 500-char chunk usually captures one complete tip or experience without cutting it in half
- 75-char overlap prevents a key piece of advice from being split across a chunk boundary (e.g., "you can appeal your financial aid — here's how" doesn't get split into two useless fragments)
- Blog posts are longer but still paragraph-structured — the recursive splitter respects those breaks

**What bad chunks look like for this corpus**:
- Too small (< 200 chars): "You should definitely appeal your" — no standalone meaning
- Too large (> 1000 chars): mixes three separate tips into one embedding, making similarity search imprecise
- No overlap: "The deadline is March 1st." in chunk 3, "Miss it and you lose priority aid." in chunk 4 — both individually useless

**Expected chunk count**: 10–12 sources × ~8 chunks each = roughly 80–120 total chunks. Well within the 50–2000 range.

## Retrieval Approach

**Embedding model**: `all-MiniLM-L6-v2` via `sentence-transformers`
- Runs locally — no API key, no rate limits, no cost
- 384-dimensional vectors — fast and lightweight
- Good semantic understanding for informal, conversational text (fits Reddit content well)

**top-k**: 4 chunks per query
- 4 chunks gives the LLM enough context to synthesize a useful answer without overwhelming it with loosely related content
- For survival tips, 4 distinct tips/experiences is usually the right response depth

**Production tradeoffs** (if this were deployed at scale):
- Context length: MiniLM is limited to 256 tokens — longer posts get truncated. For production, `text-embedding-3-small` (OpenAI) or `embed-english-v3.0` (Cohere) supports 8k tokens
- Multilingual support: MiniLM is English-only. A campus with international students would need `paraphrase-multilingual-MiniLM-L12-v2`
- Accuracy: domain-specific fine-tuned models would outperform general-purpose MiniLM, but require training data
- Latency: local models add ~50ms per query; API embeddings add network latency but offload compute

## Evaluation Plan

5 test questions with specific, verifiable expected answers:

| # | Question | Expected correct answer |
|---|----------|------------------------|
| 1 | "What should I do if my financial aid was reduced or cut?" | Appeal it in writing, provide documentation (job loss, medical bills), contact the financial aid office directly, ask about SAP appeals |
| 2 | "How do I get off a course waitlist?" | Email the professor directly, show up the first day, explain your situation, ask about add codes or department overrides |
| 3 | "What campus resources exist for students struggling with mental health?" | Counseling center (often free), crisis hotlines, peer support groups, CAPS (Counseling and Psychological Services), student wellness centers |
| 4 | "I missed the FAFSA deadline — what are my options?" | File anyway (some aid is still available), contact financial aid office, check for state/institutional deadlines separately, look for emergency funds |
| 5 | "How do I balance a part-time job with full-time classes?" | Limit to 15–20 hours/week, prioritize work-study, use a planner, talk to advisor about course load, use campus resources (tutoring, food pantry) to reduce other stressors |

## Anticipated Challenges

1. **Reddit scraping limitations**: Reddit's API now requires OAuth and rate-limits heavily. Plan B: manually copy 10–12 representative threads as `.txt` files, or use pushshift.io archives. This is expected and documented.

2. **Chunk boundary splits on key facts**: A tip like "The deadline is March 1st — miss it and you lose priority consideration" could split across chunks. Mitigation: 75-char overlap and sentence-boundary splitting.

3. **Out-of-vocabulary terms**: Abbreviations like "SAP" (Satisfactory Academic Progress), "FAFSA", "CAPS" may not embed well without context. If retrieval fails on these, we'll add a brief glossary prefix to affected chunks.

4. **Off-topic retrieval**: A query about mental health might return financial aid chunks if they share surface-level words ("stress", "help", "support"). Mitigation: tune top-k and inspect distance scores during testing.

## AI Tool Plan

| Pipeline component | What I'll give the AI | What I expect it to produce |
|---|---|---|
| Ingestion + cleaning (`ingest.py`) | This Documents section + Chunking Strategy section | Script that loads `.txt` files, strips boilerplate, returns clean strings |
| Chunking (`ingest.py`) | Chunking Strategy section with exact numbers (500 chars, 75 overlap) | `chunk_text()` function using `langchain.text_splitter.RecursiveCharacterTextSplitter` |
| Embedding + ChromaDB (`embed.py`) | Retrieval Approach section + pipeline diagram | Script that embeds all chunks with MiniLM and stores in ChromaDB with source metadata |
| Retrieval function (`query.py`) | Retrieval Approach section (top-k=4) | `retrieve(query, k=4)` function returning chunks + source names + distance scores |
| Generation (`query.py`) | Grounding requirement + system prompt design | `ask(question)` function that builds grounded prompt, calls Groq, returns answer + sources |
| Notebook interface (`notebook.ipynb`) | Interface requirement + Gradio/notebook instructions | Jupyter notebook with interactive widgets for query + response display |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OFFLINE (run once)                                         │
│                                                             │
│  Raw .txt files                                             │
│       │                                                     │
│       ▼                                                     │
│  [ingest.py] ── clean + chunk ──► chunks[]                  │
│       │         500 chars, 75 overlap                       │
│       ▼                                                     │
│  [embed.py]  ── all-MiniLM-L6-v2 ──► ChromaDB              │
│                  (sentence-transformers)   (local)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ONLINE (per query)                                         │
│                                                             │
│  User question                                              │
│       │                                                     │
│       ▼                                                     │
│  [query.py] ── embed query ──► ChromaDB semantic search     │
│                                    │ top-4 chunks           │
│                                    ▼                        │
│               build grounded prompt + retrieved context     │
│                                    │                        │
│                                    ▼                        │
│               Groq llama-3.3-70b-versatile                  │
│                                    │                        │
│                                    ▼                        │
│               Answer + source citations                     │
│                                    │                        │
│                                    ▼                        │
│  [notebook.ipynb] ── display to user                        │
└─────────────────────────────────────────────────────────────┘
```
