# Interview Bank — AI Engineer

One section per lesson. Each answer is a **model answer**: the structure an interviewer wants to hear
(definition, then why it matters, then trade-off, then an example from *your* project). Rewrite them
in your own words once you've built the step. An answer backed by something you built and measured
beats a memorized one.

Tip for every answer: **end with a concrete example from the telecom project**, ideally with a number.

---

## 00a — Python engineering toolkit

**Q1. Why is Pydantic so common in LLM applications?**
LLM apps have more untrusted boundaries than usual: user input, model output, tool-call arguments,
retrieved documents. Pydantic validates and coerces data at each boundary into typed objects, with
precise errors. It also **generates JSON Schema** from the same class, and that schema is what you send
to the model for structured outputs and tool definitions. So one class is your validator, your LLM
contract and (in FastAPI) your API docs. Field descriptions become part of the prompt, so writing them
well improves extraction accuracy.
*Project example:* "`UsageProfile` both told the model what to extract and rejected negative or absurd
values the model occasionally produced."

**Q2. BaseModel vs dataclass vs TypedDict?**
TypedDict only adds static type hints to a dict, with no runtime checks; it's good for typing existing
dict-shaped data. Dataclasses are lightweight containers with no validation, good for internal objects
you construct yourself. Pydantic validates and coerces at runtime and serializes to JSON and JSON Schema,
so use it wherever data crosses a boundary. It costs more per object, so in hot loops you can validate at
the edge and convert to plain classes inside.

**Q3. Lax vs strict?**
Lax (the default) coerces compatible types, e.g. `"300"` → `300`. That's useful for query strings and
LLM output, which sometimes quotes numbers. Strict rejects any type mismatch; use it for internal
contracts where a wrong type signals a real upstream bug you'd want surfaced rather than papered over.
You can set strictness per model or per field.

**Q4. How do you test code that calls an LLM?** *(very common)*
Three layers. (1) **Unit tests with fakes or mocks** for everything deterministic around the model: prompt
assembly, parsing, cost accounting, retries, tool dispatch. They're fast, free, and run in CI on every
commit. (2) **A few live integration tests**, marked and run on demand or nightly, to catch API contract
changes. (3) **Evals** on a golden dataset for *quality* (retrieval recall, faithfulness, tool-call
accuracy), run when prompts, models or retrieval change, with thresholds as a CI gate. Optionally,
record/replay fixtures to make integration tests deterministic. The key point is to separate
"does my code work" (tests) from "is the model's behaviour good enough" (evals).

**Q5. Linter vs formatter; why ruff?**
A linter statically finds likely bugs and bad patterns (unused imports, undefined names, mutable defaults,
bare excepts). A formatter rewrites code to a single style so reviews focus on logic. Ruff does both and
replaces flake8, black, isort and pyupgrade with one tool and one config. It's 10–100× faster, so it's
cheap enough to run on every save and every commit.

**Q6. Pre-commit and CI — why both?**
Pre-commit gives instant local feedback and stops trivial problems, especially leaked secrets, before
they ever enter history. But it's client-side and skippable (`--no-verify`, or a teammate who never
installed it). CI is the enforced, server-side gate and can run heavier checks (full test suite, evals,
Docker build). The same tool versions in both keep them consistent.

**Q7. What's a pytest fixture?**
A function that provides setup (and optional teardown) to tests that request it by argument name. It's
reusable, composable, and scoped (function, module, session). Built-ins include `tmp_path` (fresh
directory) and `monkeypatch` (temporarily replace attributes or env vars).
*Project example:* "My `llm` fixture builds an `LLMClient` with a fake API client and redirects the call
log to `tmp_path`, so tests are offline, free and leave no files behind."

**Q8. Keeping secrets out of the repo; what if one leaks?**
Prevention: secrets in env vars or a secret manager, never in code; `.env` gitignored with a committed
`.env.example`; pre-commit secret scanning; CI secret scanning (e.g. GitHub push protection). If one leaks:
**rotate or revoke the key immediately** (that comes first, because deleting the commit doesn't help once
it's been pushed or cloned), check usage logs for abuse, then purge it from history and add a check so
it can't happen again. Spending limits on API accounts reduce the damage.

---

## 00b — LLM API fundamentals

**Q1. What is a token, and why charge by token?**
A token is a sub-word unit produced by the model's tokenizer (BPE). In English it's roughly 4 characters,
and numbers, code and non-Latin scripts take more. Compute cost scales with tokens: every input token
goes through the forward pass, and every output token needs its own sequential decode step. So
per-token pricing tracks the provider's real cost, unlike per-request pricing.
*Project example:* "I measured it instead of assuming: with the current o200k tokenizer, an Arabic usage
sentence was 14 tokens vs 15 in English, so the old 'non-English costs 2–4× more' rule didn't hold. The
bigger surprise was structured output: a 15-token message was billed about 270 input tokens, because the
JSON Schema is sent with every call."

**Q2. Why is output pricier and slower than input?**
Input is processed in one parallel pass (prefill). Output is autoregressive: one token at a time, each
step reading the whole KV cache, so it's memory-bandwidth bound and sequential. Latency is therefore
dominated by output length. Levers: cap `max_output_tokens`, ask for concise answers, stream to cut
*perceived* latency (time to first token).

**Q3. Huge context window — why still RAG?**
(1) Cost: you pay for every input token on every query. (2) Accuracy: "lost in the middle", recall
degrades for facts buried in long contexts, and irrelevant text distracts the model. (3) Latency grows
with prompt size. (4) Freshness and access control: a retrieval index updates per document and can
filter by user permissions. (5) Citations come naturally. Long context *is* useful when the relevant set
is small but must be read in full (e.g. one contract). Good answer: "RAG to select, long context to
reason over what was selected."

**Q4. Is temperature 0 deterministic?**
No. Floating-point non-associativity on GPUs, batching with other requests, and silent model updates
all cause variation. Implications: evaluate over a set and report averages or pass rates; for critical
cases run several times and measure consistency; pin model versions; keep logic that must be exact
(prices, eligibility) in deterministic code, not in the LLM.

**Q5. Reliably parseable output?**
Use schema-constrained structured outputs (JSON Schema or Pydantic), where decoding can only produce
schema-valid tokens. That's better than JSON mode (valid syntax, any shape), which is better than
prompt-only. What can still go wrong: semantically wrong values, hallucinated values for fields the
input doesn't support (fix: make them optional or nullable and tell the model to use null), refusals,
truncation at the max-token limit. So validate business rules after parsing and evaluate extraction
accuracy on a labelled set.

**Q6. Cost estimate: 50k queries/day, 2.5k in / 250 out, $0.40 / $1.60 per M.**
Per day: input 125M tokens × $0.40 = $50; output 12.5M × $1.60 = $20; total $70/day, about $2,100/month.
Then mention the levers: prompt caching on the static prefix, fewer and shorter retrieved chunks,
routing easy queries to a smaller model, caching answers to repeated FAQ questions, batch API for
offline jobs. *Showing the arithmetic and then the optimization levers is what scores.*

**Q7. Prompt caching?**
The provider reuses the computed attention state (KV cache) for a prompt prefix that matches a recent
request, and bills those tokens at a discount with lower latency. It only works on an *exact prefix*,
so put stable content first (system prompt, tool schemas, few-shot examples) and variable content last
(retrieved chunks, user question, timestamps). A common bug: putting the current date or a request ID
at the top of the system prompt, which breaks caching for every call.

**Q8. Stateless API — how does a chatbot remember?**
The client resends the conversation history every turn. Problems: cost and latency grow every turn,
the context window can overflow, and old irrelevant turns distract the model. Strategies: sliding
window of the last N turns, summarizing older turns, storing facts as structured memory, and for RAG,
**rewriting the follow-up question into a standalone query** before retrieval ("what about roaming?"
→ "roaming charges for the Super 50 plan").

---

## 01 — Data for RAG

**Q1. Which questions go to documents, which to structured data?**
Exact facts, numbers, filters, comparisons and calculations ("cheapest plan with 20 GB", "my balance")
go to structured data via code: SQL, an API, or a tool the LLM calls. Explanations, policies and
how-tos go to RAG. LLMs are unreliable at sorting, filtering and arithmetic over retrieved text, and code
never is. Many production assistants are a hybrid: an LLM router plus tools plus retrieval.
*Project example:* "Plan prices live in a JSON catalog that the tools query. The plan pages for RAG are
rendered from the same JSON, so the two can't drift apart."

**Q2. Why does metadata matter in RAG?**
(1) Filtering: market, product, language, date, and **access control** (users only retrieve documents
they're allowed to see). (2) Citations and traceability: which source an answer came from. (3) Freshness:
effective dates and versions to exclude or down-rank stale content. (4) Debugging and evals: knowing
what was retrieved and why, and labelling ground truth by doc_id. Capture it at ingestion, because
recovering it later from chunks is painful.

**Q3. Outdated or conflicting documents?**
First, version documents with effective dates and `superseded_by` links. Then pick a strategy: filter to
current versions by default; down-rank older ones; or pass dates into the prompt and instruct the model
to prefer the most recent version and mention the change. Keep history for "what was my old price"
questions. And **measure** it: have an eval category for stale-document leakage.
*Project example:* "I deliberately kept the 2025 roaming policy in the corpus and tracked how often its
old rates leaked into answers, before and after adding date filtering."

**Q4. Synthetic data: pros and cons?**
Pros: exact ground truth, designed edge cases (traps), no legal or privacy issues, stable and versioned.
Cons: less realistic and less varied than real data, LLM-written text sounds alike, the LLM can invent
facts while generating, and eval leakage if you generate questions from the same text with the same
model. Mitigations: generate from fixed facts with automated checks, vary styles, write some eval
questions by hand, and validate on real data as soon as you can get it.

**Q5. What makes ingesting real enterprise documents hard?**
Parsing: PDFs with tables, multi-column layouts and repeated headers or footers; scanned pages that need
OCR; HTML boilerplate; slides with text in images. Also duplicates and near-duplicates, many versions of
the same document, missing metadata, permissions, and PII. Tools (`unstructured`, `docling`, PyMuPDF)
help, but you still have to inspect samples by hand and measure parsing quality. Tables often need
special handling, e.g. converting them to markdown or into structured rows.

**Q6. How do you stop a support bot from quoting wrong prices?**
Defense in depth. (1) Data design: prices live in structured data, the single source of truth. (2) Tools:
the model gets prices from a tool result instead of recalling them. (3) Retrieval: filter by market and
to current versions. (4) Prompting: tell the model to quote prices only from the provided context, and
to cite sources. (5) Output checks: verify that every number in the answer appears in the retrieved
context or tool results. (6) Evals: a price-accuracy category in CI.
*Project example:* "My data generator rejected documents containing numbers not in their facts. The same
idea later became an output guardrail."

**Q7. PII in a RAG corpus?**
Minimize: don't ingest what you don't need. Detect and mask or redact PII before embedding (NER-based
tools such as Presidio). Enforce access control with metadata filters at query time. Remember that
embeddings are not anonymization, because text can be partly reconstructed from vectors. Also handle
retention and deletion requests: you need to know which chunks came from which source so you can
delete them.

**Q8. Documents change daily. How do you keep the index fresh?**
Incremental indexing: track a content hash and timestamp per document, and re-chunk and re-embed only
documents that changed. Delete chunks of removed documents. Run it on a schedule or trigger it from
events (webhooks or change-data-capture). Version the index so you can roll back. Monitor freshness
(age of newest document) and run the eval set after re-indexing. Changing the *embedding model* means
re-embedding everything, so plan for that.

---

## 02 — Chunking

**Q1. Why chunk instead of embedding whole documents?**
One vector per long document is a diluted average of all its topics, so it matches no specific
question strongly. Chunks give sharp embeddings, send the LLM only relevant text (fewer tokens, less
distraction), fit embedding models' input limits (text beyond them is silently truncated), and allow
precise citations. The trade-off is context: chunks that are too small lose their subject.

**Q2. How do you choose chunk size?**
Start from the documents and the questions: short factual questions favour smaller chunks, questions
needing broad context favour larger ones. Typical starting points are 200–500 tokens, then **measure
recall@k and answer quality** on an eval set across a few sizes. Too small: facts lose their subject
("it costs 25 SAR"), and you need more chunks per query. Too large: diluted embeddings, noise, and
more tokens per query.
*Project example:* "Our documents are short and structured, so section-level chunks with a 300-token
cap kept 100% of facts whole while embedding the fewest tokens. I validated it against a fixed-size
baseline on recall@k."

**Q3. What is overlap and what does it cost?**
Repeating the last N tokens of each chunk at the start of the next, so content cut at a boundary
appears whole somewhere. Costs: more tokens to embed and store (+15% in our corpus at 50/200), and
near-duplicate results that crowd the top-k. It patches blind cutting. Structure-aware boundaries
mostly remove the need for it.

**Q4. Fixed vs recursive vs structure-aware vs semantic?**
Fixed: every N tokens; simple, but cuts sentences and facts. Recursive: splits on paragraphs, then
sentences, then characters until pieces fit; the common default. Structure-aware: uses headings, HTML
or PDF sections, so chunks match the author's topics and headings provide context; needs structured
input. Semantic: starts a new chunk where sentence embeddings shift in meaning; useful for
unstructured prose, but costs more and gains are often modest. Choose based on the documents, then
validate with evals.

**Q5. Contextual headers / contextual retrieval?**
Chunks lose context once cut out ("the fee is 30 AED": which fee, which market?). Contextual headers
prepend the document title and section path before embedding. Anthropic's Contextual Retrieval uses an
LLM to write a 1–2 sentence context for each chunk; they reported large reductions in retrieval
failures, especially combined with BM25 and reranking. The cost is an LLM call per chunk at indexing
time, reduced with prompt caching.
*Project example:* "Each chunk is embedded as 'Roaming Policy (KSA) > GCC Day Pass' plus its text, so
even a one-line FAQ answer carries its subject and market."

**Q6. Parent-child (small-to-big) retrieval?**
Index small chunks for precise matching, but return their larger parent (section or document) to the
LLM. You get the precision of small chunks with the context of large ones. Variants: sentence-window
retrieval (return neighbouring sentences), and auto-merging (if several children of one parent are
retrieved, return the parent instead).

**Q7. Chunking FAQs, tables, code?**
FAQs: one question-and-answer pair per chunk; the question itself is an excellent retrieval key. Tables:
keep small tables whole; for big ones, one row (or a group of rows) per chunk with the column headers
repeated, or convert them to text sentences; never cut mid-table. Code: split by function or class
(AST-aware), with the file path and signature as context.

**Q8. How do you evaluate a chunking strategy?**
Ultimately by downstream metrics: retrieval recall@k and MRR on a golden question set, then answer
faithfulness and correctness. Cheap proxies before that: fact integrity (do known facts survive in one
chunk?), size distribution (tiny or huge chunks), and reading samples. Compare strategies with
everything else held fixed (same embedding model, same k).
