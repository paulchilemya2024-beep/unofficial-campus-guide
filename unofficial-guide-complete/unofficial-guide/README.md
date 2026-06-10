# The Unofficial Campus Survival Guide
**AI201 Project 1 — RAG System**

A Retrieval-Augmented Generation (RAG) system that makes student-generated campus survival knowledge searchable and answerable. Students ask plain-language questions about registration, financial aid, mental health, and campus resources — and get grounded, cited answers drawn from real student-written documents.

---

## Domain and Document Sources

**Domain**: General college survival — the practical knowledge students pass to each other that official university websites don't capture. What actually happens when you miss a FAFSA deadline. How to appeal a financial aid cut. Which campus mental health resources have the shortest wait times. How to recover from academic probation. This knowledge lives in Reddit threads, student forums, and peer conversations — not the registrar's website.

**Why it's hard to find through official channels**: Universities publish polished institutional language. The real student experience — including what to do when things go wrong — only exists in peer-to-peer sources.

**Document sources** (12 total, stored in `docs/`):

| File | Source | Topic |
|------|--------|-------|
| `01_financial_aid_appeal.txt` | r/college | How to appeal a financial aid reduction |
| `02_fafsa_tips.txt` | r/financialaid | FAFSA filing strategy and common mistakes |
| `03_registration_waitlists.txt` | r/college | Getting off course waitlists |
| `04_mental_health_resources.txt` | r/college | Campus counseling and crisis resources |
| `05_academic_probation.txt` | r/college | Academic probation recovery |
| `06_student_loans.txt` | r/StudentLoans | Loan types, IDR plans, PSLF |
| `07_work_study_parttime.txt` | r/college | Balancing work and school |
| `08_campus_food_emergency_funds.txt` | r/college | Food pantries and emergency grants |
| `09_first_gen_tips.txt` | r/college | First-generation student survival |
| `10_registration_strategies.txt` | r/college | Registration timing and strategy |
| `11_grade_appeals.txt` | r/college | How to formally appeal a grade |
| `12_campus_health_services.txt` | r/college | Campus health center services |

---

## Chunking Strategy and Reasoning

**Strategy**: Recursive paragraph-first splitting with hard character fallback.

**Parameters**:
- Chunk size: **500 characters** (~80–100 words)
- Overlap: **75 characters** (~15% of chunk size)

**Why these fit the documents**: Reddit comments and student posts are short and opinionated — one tip or experience per post. A 500-character chunk captures one complete thought without cutting it. The 75-character overlap prevents key advice from being split across chunk boundaries (e.g., "the deadline is March 1st" ending one chunk while "miss it and you lose priority aid" starts the next). The recursive splitter tries paragraph breaks first, then sentence breaks, then hard character cuts — so natural boundaries are respected wherever possible.

**Sample chunks** (5 labeled examples):

```
[Chunk 1 — 01_financial_aid_appeal.txt, index 0]
"So my financial aid was cut by $4,000 this year and I was completely blindsided.
Here's what worked for me when I appealed: First, write a formal appeal letter. Be
specific — don't just say 'I need more money.' Explain exactly what changed in your
financial situation. I listed my dad's job loss, the medical bills from my mom's
surgery, and attached documentation for all of it."
[498 chars]

[Chunk 2 — 02_fafsa_tips.txt, index 1]
"FAFSA opens October 1st every year. File it THE DAY IT OPENS. Financial aid is
first-come, first-served for most institutional grants. Schools give out their own
grant money until it's gone. If you file in February instead of October you might
get the same federal aid but miss out on thousands in school-specific grants."
[327 chars]

[Chunk 3 — 04_mental_health_resources.txt, index 0]
"Your campus counseling center (often called CAPS — Counseling and Psychological
Services) almost always offers free sessions for currently enrolled students. Wait
times can be long (2–4 weeks for a non-urgent appointment) so schedule earlier
than you think you need to. Many schools offer same-day 'urgent' appointments if
you're in crisis — call and explicitly say you need urgent support."
[394 chars]

[Chunk 4 — 08_campus_food_emergency_funds.txt, index 1]
"Emergency funds and emergency grants: Most schools have a small fund for students
facing sudden financial hardship — unexpected medical bills, car repairs, sudden
loss of income. The amounts vary from $200 to $2000 but can be life-saving. Email
your Dean of Students office and ask specifically about emergency student funds.
You won't be penalized for asking."
[357 chars]

[Chunk 5 — 09_first_gen_tips.txt, index 2]
"Go to office hours. First-gen students often avoid office hours because they feel
like it's showing weakness. The opposite is true. Students who go to office hours
get better grades, better recommendations, and better research opportunities.
Professors remember names. Names get opportunities."
[291 chars]
```

---

## Embedding Model and Retrieval

**Embedding model**: `all-MiniLM-L6-v2` via `sentence-transformers`

**Why this model**: Runs entirely locally — no API key, no rate limits, no cost. Produces 384-dimensional vectors with strong semantic understanding for conversational, informal text (which matches the Reddit corpus well). Fast enough to embed all 90+ chunks in under 30 seconds on a laptop CPU.

**Production tradeoffs**:
- *Context length*: MiniLM truncates at 256 tokens — long posts get cut off. For production, `text-embedding-3-small` (OpenAI, 8k tokens) or `embed-english-v3.0` (Cohere) would handle longer documents.
- *Multilingual support*: MiniLM is English-only. A campus with international students would need `paraphrase-multilingual-MiniLM-L12-v2`.
- *Domain accuracy*: A model fine-tuned on student text would outperform a general-purpose model, but requires labeled training data.
- *Latency vs. cost*: Local models add ~50ms per query with zero cost; API embeddings add network latency but offload compute for high-traffic deployments.

**top-k**: 4 chunks per query. Four distinct student tips/experiences is the right depth for survival guide answers — enough context for synthesis without diluting relevance.

---

## Retrieval Test Results

**Query 1**: "What should I do if my financial aid was cut?"

Top chunks retrieved:
1. `01_financial_aid_appeal.txt` (distance: 0.21) — describes writing formal appeal, attaching documentation, requesting professional judgment review
2. `02_fafsa_tips.txt` (distance: 0.38) — mentions special circumstances adjustment for income changes
3. `08_campus_food_emergency_funds.txt` (distance: 0.41) — describes emergency grant funds from Dean of Students
4. `01_financial_aid_appeal.txt` (distance: 0.44) — SAP appeal process for GPA-based cuts

*Why these chunks are relevant*: The top result is a direct match — financial aid appeal advice. The second and third results correctly surface related survival strategies (income adjustments, emergency funds) that a student in this situation would actually need. All four distances are well below 0.5.

---

**Query 2**: "What mental health resources does my campus have?"

Top chunks retrieved:
1. `04_mental_health_resources.txt` (distance: 0.18) — CAPS, free sessions, urgent appointments
2. `04_mental_health_resources.txt` (distance: 0.22) — 988 lifeline, Crisis Text Line, peer counseling
3. `12_campus_health_services.txt` (distance: 0.35) — campus health mental health referrals, bridge prescriptions
4. `04_mental_health_resources.txt` (distance: 0.39) — off-campus therapist options via insurance

*Why relevant*: The dedicated mental health document dominates, which is correct. The campus health chunk adds useful complementary information (bridge prescriptions, insurance navigation). Strong retrieval signal.

---

**Query 3**: "How do I balance work and school without burning out?"

Top chunks retrieved:
1. `07_work_study_parttime.txt` (distance: 0.29) — 15-20 hour cap, work-study benefits
2. `07_work_study_parttime.txt` (distance: 0.33) — time block scheduling, campus resources
3. `08_campus_food_emergency_funds.txt` (distance: 0.51) — food pantry, emergency funds to reduce work need
4. `09_first_gen_tips.txt` (distance: 0.58) — general ask-for-help advice

*Note*: Chunks 3 and 4 have higher distances (0.51, 0.58). They're tangentially relevant but not directly on-topic. This is a borderline retrieval — see Failure Case below.

---

## Grounded Response Generation

**How grounding is enforced**: The LLM receives a system prompt that explicitly states: *"Answer ONLY using the information in the provided documents below. Do NOT use your general training knowledge to fill in gaps."* The temperature is set to 0.3 to reduce creative deviation from the retrieved context. If no relevant chunks exist, the system prompt instructs the model to explicitly say it doesn't have enough information rather than generating from general knowledge.

**Source attribution**: Source filenames are extracted programmatically from the retrieved chunks *before* generation and appended to the response object. This guarantees attribution even if the LLM forgets to cite sources in its text.

**Example response 1 (grounded, in-scope)**:

> *Question: What should I do if my financial aid was reduced?*
>
> Based on student experiences, here's what you should do if your financial aid was cut:
>
> First, write a formal appeal letter with specific documentation — job loss letters, medical bills, anything concrete. The financial aid office needs evidence, not just a statement of need. Specifically request a "professional judgment" review, which allows a financial aid officer to override your Expected Family Contribution based on special circumstances. Many students don't know this option exists.
>
> While your appeal is being reviewed, ask about emergency student funds through your Dean of Students office. These grants ($200–$2,000) can bridge the gap while the appeal is processed.
>
> Be persistent with follow-up — multiple polite emails with documentation attached. The process can take 4–6 weeks.
>
> Sources: [01_financial_aid_appeal.txt, 08_campus_food_emergency_funds.txt]

**Example response 2 (out-of-scope refusal)**:

> *Question: What is the best programming language for machine learning?*
>
> I don't have enough information on that in my sources. Try asking your school's Dean of Students office or r/college.

---

## Query Interface

**Interface type**: Jupyter notebook with `ipywidgets` interactive components.

**Inputs**: Text area for free-form question entry. Five pre-populated example buttons that auto-fill common survival questions.

**Outputs**:
- Answer section with the grounded LLM response
- "Retrieved from" section listing source document filenames
- Expandable "Retrieved chunks" section showing the exact text and distance score for each of the 4 retrieved chunks (for transparency and debugging)

**Sample interaction transcript**:

```
Input: "I missed the FAFSA deadline — what are my options?"

Output:
  📖 Answer
  Even if you've missed the federal priority deadline, you have several options.
  First, file the FAFSA anyway — the federal deadline is June 30th of the award
  year, and you may still qualify for federal loans and some grants. Schools have
  their own priority deadlines (often February or March) that are separate from
  the federal deadline, so contact your financial aid office to understand your
  specific situation.

  If your family's income changed significantly from the tax year on the form,
  contact the financial aid office about a "special circumstances" adjustment
  using current-year income. Also ask about emergency student funds through
  your Dean of Students office, which can provide bridge support while financial
  aid is being processed.

  Sources: [02_fafsa_tips.txt, 08_campus_food_emergency_funds.txt]

  📎 Retrieved from:
  • 02_fafsa_tips.txt
  • 08_campus_food_emergency_funds.txt

  🔍 Retrieved chunks (expand to inspect)
  [Chunk 1: 02_fafsa_tips.txt — distance: 0.24] ▶
  [Chunk 2: 02_fafsa_tips.txt — distance: 0.31] ▶
  [Chunk 3: 08_campus_food_emergency_funds.txt — distance: 0.44] ▶
  [Chunk 4: 01_financial_aid_appeal.txt — distance: 0.49] ▶
```

---

## Evaluation Report

| # | Question | Expected Answer | System Accuracy |
|---|----------|----------------|-----------------|
| 1 | What should I do if my financial aid was reduced? | Appeal with documentation, request professional judgment, ask about emergency funds | Accurate |
| 2 | How do I get off a course waitlist? | Email professor, show up first day, ask for department override, watch for drops | Accurate |
| 3 | What campus mental health resources exist? | CAPS (free), 988, Crisis Text Line, peer counseling, wellness center | Accurate |
| 4 | I missed the FAFSA deadline — what are my options? | File anyway, contact FA office, check institutional deadlines, look for emergency funds | Accurate |
| 5 | How do I balance a part-time job with full-time classes? | Cap at 15-20 hrs, prioritize work-study, time block schedule, use campus resources | Partially accurate |

### Failure Case

**Question 5**: "How do I balance a part-time job with full-time classes?"

**System response summary**: Correctly cited the 15-20 hour cap and work-study recommendation, but the advice to "use campus food pantry and emergency funds to reduce financial pressure so you can work fewer hours" was vague and under-explained. The connection between reducing campus expenses and needing fewer work hours — which is a key insight in the source document — was not clearly communicated.

**Specific cause**: The third and fourth retrieved chunks (distances 0.51 and 0.58) came from `08_campus_food_emergency_funds.txt` and `09_first_gen_tips.txt` — tangentially relevant but not directly about work-life balance. The LLM tried to synthesize all four chunks and the result was diluted. Reducing top-k from 4 to 3 for this query type, or filtering to return only chunks with distance < 0.5, would likely improve this result.

---

## Spec Reflection

**How the spec helped**: Writing the chunking strategy section before building the pipeline forced a concrete decision (500 chars, 75 overlap) grounded in actual document structure. When retrieval worked well, it was because the chunk size matched the natural length of a Reddit tip. Planning first meant the implementation had a clear target.

**How implementation diverged**: The spec called for using `langchain.text_splitter.RecursiveCharacterTextSplitter`. During implementation, the `langchain` dependency added 15+ transitive packages and a deprecation warning. Replaced with a pure-Python recursive splitter that uses the same algorithm — paragraph breaks, then sentence breaks, then hard character split — with no additional dependencies.

---

## AI Usage

1. **Generating `ingest.py` chunk_text() function**: Provided the Chunking Strategy section from planning.md (500 chars, 75 overlap, recursive paragraph-first). Claude generated an initial version using `langchain`. Reviewed and replaced with a pure-Python implementation to eliminate the dependency. The core logic (paragraph split → accumulate → overflow → hard split) was correct; the library choice was changed.

2. **Generating the system prompt in `query.py`**: Provided the grounding requirement ("only from retrieved context") and the refusal format ("I don't have enough information"). Claude generated a longer, more elaborate system prompt. Trimmed it significantly — removed explanatory preamble and reduced examples, keeping only the rules. Shorter, stricter system prompts produce more faithful grounded responses in practice.
