# Lesson 02 — Chunking

> **Goal:** understand why documents are split into chunks, the main strategies and their
> trade-offs, and build the structure-aware chunker this project uses.
> **Time:** ~3–4 hours (1.5 h reading, 1 h building, 30 min comparing, 30 min quiz).

Chunking is the first step of the RAG pipeline (**chunk** → embed → store → retrieve → generate),
and it quietly caps everything after it. If a fact is cut in half, or buried in a chunk full of
unrelated text, no embedding model or reranker can fully recover it.

---

## Part A — Concepts

### 1. Why chunk at all?

Why not embed each whole document and retrieve whole documents?

1. **One vector per document blurs the meaning.** An embedding squeezes a text into one fixed-size
   vector (say 384 or 1536 numbers). For a 2,000-word T&C covering 14 topics, that vector is an
   *average* of all of them, so it's not a strong match for any single question. A 120-word chunk
   about billing disputes is a sharp match for "how long do I have to dispute a bill?"
2. **Precision of what reaches the LLM.** Retrieval hands the top-k results to the LLM. Whole
   documents bring mostly irrelevant text: more tokens (cost, latency), and more distraction
   ("lost in the middle", Lesson 00b).
3. **Input limits.** Embedding models have a maximum input length (often 512 tokens for small
   open-source models, about 8k for OpenAI's). Text past the limit is silently cut off.
4. **Citations.** "Source: T&C, section *Charges and payments*" is far more useful than "Source: T&C".

But chunks that are **too small** lose context: "It costs 25 SAR." *What* costs 25 SAR, in which
market? So chunking is a balance between **precision** (small, focused) and **context**
(enough surrounding information to make sense alone).

### 2. Chunk size and overlap

| | Small chunks (~50–150 tokens) | Large chunks (~500–1,000 tokens) |
|---|---|---|
| Embedding | Sharp, specific | Diluted average |
| Retrieval | Precise, but may miss context | Broad, more noise |
| Tokens sent to LLM | Few per chunk; may need higher k | Many, costly |
| Risk | Fact separated from its subject | Relevant sentence drowned in noise |

There's no universal best size. It depends on the documents and the questions, which is why we
**measure** in Phase 3 instead of trusting a blog post. Common starting points: 200–500 tokens.

**Overlap** repeats the last N tokens of one chunk at the start of the next, so a sentence cut at a
boundary appears whole in at least one chunk. It costs more tokens (more to embed and store) and
produces near-duplicate results that waste slots in your top-k. It's a patch for blind cutting.
Better boundaries make it largely unnecessary.

### 3. The main strategies

| Strategy | How it splits | Pros | Cons |
|---|---|---|---|
| **Fixed-size** | Every N tokens or characters | Trivial, predictable sizes | Cuts mid-sentence and mid-fact; ignores structure |
| **Recursive** | Tries paragraph breaks first, then sentences, then characters, until pieces fit | Respects natural boundaries; the most common default (LangChain's `RecursiveCharacterTextSplitter`) | Still blind to the document's sections |
| **Structure-aware** | Uses the document's own structure: markdown headings, HTML tags, PDF sections | Chunks match how the author organized the topics; headings give free context | Needs structured input; section sizes vary a lot |
| **Semantic** | Embeds each sentence and starts a new chunk where consecutive sentences' meaning shifts | Finds topic boundaries in unstructured text | Extra embedding cost, slower, harder to debug, and gains are often small |
| **No chunking** | The whole document is one chunk | Perfect context | Only works for short documents |

Two refinements that work with any strategy:
- **Contextual headers.** Prepend the document title and section heading to each chunk before
  embedding: `"Roaming Policy (KSA) > GCC Day Pass\n\nThe pass costs 25 SAR…"`. Now "It costs 25 SAR"
  knows what *it* is and which market it's for. Anthropic's "Contextual Retrieval" generalizes this
  by using an LLM to write a short context sentence for every chunk.
- **Parent-child ("small-to-big").** Search over small chunks for precision, but send the LLM the
  larger *parent* section or document for context. You get precise retrieval and complete context.

**Special content:** FAQs chunk naturally as one question-and-answer pair each. Tables should stay
whole, or be converted to one row per chunk with the column headers repeated. Code splits by
function. Never cut a table or a Q&A pair in half.

### 4. Our decision

Our documents are **markdown with clear headings**, written by us. So:
- **Structure-aware chunking by headings.** Each section is a chunk, and each FAQ question with
  its answer is a chunk. Plan pages have no sub-headings, so each becomes one chunk.
- **A size cap (`max_tokens=300`)** with a recursive fallback (`split_long`) for sections that are
  too long.
- **Contextual headers** (title > heading) in the embedded text.
- **Every chunk inherits the document's metadata** (market, dates, `doc_type`…), so retrieval can
  filter on it in Lesson 09.
- **Baseline for Phase 3:** fixed-size 200 tokens with 50 overlap. We'll compare both on recall@k,
  so the decision is backed by numbers, not just reasoning.

### 5. How do you know chunking is good?

The real answer: **retrieval evals** (Phase 3). Does the right chunk come back for each question?
But some cheap checks catch obvious problems now:
- **Fact integrity.** Do our ground-truth facts survive in one piece? We use each fact's numbers as a
  fingerprint: if no single chunk contains all of "25 SAR … 1 GB … 24 hours", the fact was cut.
- **Size distribution.** Tiny chunks (<30 tokens) are often useless alone, and huge ones are diluted.
- **Read some chunks.** As in Lesson 01, eyeballing beats any metric for spotting nonsense.

---

## Part B — Build

**You write one function**, the core of this lesson. I wrote the rest (`Chunk` model, fixed-size
baseline, `split_long`, metrics, comparison script, tests).

### Task 1 — `chunk_by_headings()` in `src/telco_support_agent/chunking.py` (~1 h)
The rules are in the docstring, and `tests/test_chunking.py` checks each one. In short:
1. Walk the lines of `doc.text`. A line starting with `# `, `## ` or `### ` begins a new section.
2. Collect each section as (heading, body). The `# Title` line is not content.
3. Skip sections with an empty body.
4. For each section, `split_long(body, max_tokens)` gives one or more pieces. Each piece becomes a
   chunk via `make_chunk(doc, n, piece, heading, context)`, with context `"<title> > <heading>"`.

Hints: `re.match(r"^(#{1,3}) (.*)", line)` captures the hashes and the heading text. Keep a running
counter for `n` across all sections.

Run `uv run pytest tests/test_chunking.py -v` until it's green.

### Task 2 — Compare strategies (~30 min)
```bash
uv run python scripts/compare_chunking.py
uv run python scripts/compare_chunking.py --show terms-and-conditions headings-300
uv run python scripts/compare_chunking.py --show terms-and-conditions fixed-200
uv run python scripts/compare_chunking.py --show faq-roaming headings-300
```
Before opening the fold below, **predict**: which strategy splits the most facts? Which makes the
most tiny chunks? Why might fixed-500 look "perfect"?

<details>
<summary>Expected results and how to read them (open after running)</summary>

| strategy | chunks | mean tokens | tiny | total tokens | facts intact |
|---|---|---|---|---|---|
| fixed-100 | 152 | 84 | 16 | 12,707 | 97.2% |
| fixed-200 | 97 | 131 | 1 | 12,703 | 97.2% |
| fixed-200-overlap-50 | 101 | 145 | 0 | 14,654 (+15%) | 100% |
| fixed-500 | 65 | 195 | 0 | 12,704 | 100% |
| headings-300 | 201 | 55 | 51 | 11,101 | 100% |

- **Fixed-size cuts facts** at window edges, e.g. *"The GCC Day Pass costs 35 SAR for 500 MB… valid
  for 24 hours"* split across two chunks. A question about it retrieves half the fact.
- **Overlap fixes the cutting, at a price:** 15% more tokens and near-duplicate chunks.
- **fixed-500 looks perfect but isn't.** Most of our documents are under 500 tokens, so it's really
  "one chunk per document": nothing gets cut, but the T&C becomes 3–4 big diluted chunks. The proxy
  metric can't see dilution. That's exactly why we need retrieval evals.
- **headings-300 keeps every fact whole and embeds the fewest tokens** (no overlap waste), but makes
  **51 tiny chunks**: short FAQ answers and one-paragraph sections. The context header gives each one
  its subject. Whether tiny chunks hurt retrieval is a question for Phase 3 (an option then: merge
  sections under ~30 tokens with their neighbour).
- **Our corpus is small and clean, so the differences are modest.** On real 50-page PDFs, the gap
  between blind and structure-aware chunking is much larger.
</details>

When you're done, commit and say "lesson 02 done". I'll clean up, then quiz you.

---

## Part C — Interview drill

Short-format quiz on "review lesson 02". Full model answers:
[interview-bank §02](../interview-bank.md#02--chunking).

1. Why chunk documents instead of embedding them whole?
2. How do you choose chunk size? What happens if chunks are too small or too large?
3. What is chunk overlap, and what does it cost?
4. Compare fixed-size, recursive, structure-aware and semantic chunking.
5. What are contextual headers / contextual retrieval?
6. What is parent-child (small-to-big) retrieval?
7. How would you chunk FAQs, tables and code?
8. How do you evaluate a chunking strategy?
