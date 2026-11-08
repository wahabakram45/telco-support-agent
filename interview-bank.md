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

---

## 03 — Embeddings & vector search

**Q1. What is an embedding, and how are embedding models trained?**
A fixed-length vector representing a text's meaning, so similar meanings are close in vector space.
Models are transformer encoders trained contrastively on (query, relevant passage) pairs: matching
pairs are pulled together, in-batch negatives pushed apart; pooling turns token vectors into one
vector. It's a bi-encoder: query and documents are encoded separately, so document vectors are
precomputed and search is fast. A cross-encoder reads query and document together: more accurate,
too slow for full search, so it's used to rerank the top results.
*Project example:* "'Phone internet on holiday in Dubai' retrieved our UAE roaming policy with no
shared keywords."

**Q2. Cosine vs dot product vs Euclidean?**
Cosine measures angle only; dot product also rewards vector length; Euclidean is straight-line
distance. For unit-length vectors all three give the same ranking (‖a−b‖² = 2 − 2cos). So normalize
once at index time, and search becomes one matrix-vector product. Use whatever the model was trained
with; cosine is the default for text.

**Q3. How do you choose an embedding model?**
Shortlist from the MTEB retrieval leaderboard, then decide on your own eval set: recall@k on real
questions. Also weigh language coverage (Arabic for us), dimensions (storage and speed; Matryoshka
models can be truncated), max input length, cost and latency, and privacy (API vs self-hosted
open models such as BGE or multilingual-E5). Changing later means re-embedding everything, so choose
deliberately.
*Project example:* "text-embedding-3-small: $0.0003 to embed the whole corpus, decent Arabic; I'd
only move to -large if evals show retrieval is the bottleneck."

**Q4. What happens when you switch embedding models in production?**
Old and new vectors are in incompatible spaces, so every document must be re-embedded, and queries
must switch at the same moment. Do it blue-green: build a new index alongside the old one, run the
eval set on both, switch traffic atomically, keep the old one for rollback. Store the model name
and version with the index and check it at query time.
*Project example:* "Our retriever refuses to start if the index's model differs from the query
model, because mixing them fails silently."

**Q5. Exact vs approximate search; how does HNSW work?**
Exact search compares the query with every vector, O(n·d): fine up to around 100k vectors, and the
ground truth. ANN trades a little recall for big speedups. HNSW builds a layered proximity graph:
search enters at a sparse top layer, greedily moves toward the query, and descends to denser layers.
Knobs: M (links per node: memory vs recall), ef_construction (build quality), ef_search (query
recall vs latency). Alternatives: IVF (search only the nearest k-means clusters, tuned by nprobe)
and PQ (compressed vectors). Evaluate with recall@k against exact search, plus latency and memory.

**Q6. Which vector database would you choose?**
It depends on scale and the existing stack. Prototype: numpy, FAISS or Chroma. If the company
already runs Postgres: pgvector, with one system, SQL joins to business data, and transactions; it's
enough for millions of vectors. Very large scale, heavy filtering or built-in hybrid search: Qdrant,
Weaviate or Milvus. No operations team: managed Pinecone. What matters more than the brand:
metadata filtering, updates and deletes, hybrid search, and operational fit.

**Q7. Pre-filtering vs post-filtering?**
Pre-filtering restricts the candidates by metadata first, then ranks them, so you always get k valid
results. Post-filtering takes the top-k, then drops mismatches, and can return few or none. With
ANN graphs, restrictive pre-filters can hurt recall unless the DB supports filtered HNSW. Use
filters for facts embeddings can't distinguish.
*Project example:* "'KSA roaming prices' vs 'UAE roaming prices' scored 0.755 similarity, so market
and freshness are hard filters, not left to the embedding."

**Q8. Where does dense retrieval fail, and what do you do?**
Negation ("cancel" vs "don't cancel" scored 0.82), numbers (55 vs 95 SAR scored 0.93), exact IDs and
rare terms (shaky margins), and out-of-domain questions: top-k always returns something, and scores
overlap with real but hard questions. Fixes: hybrid search with BM25 for exact terms, a reranker,
metadata filters, structured tools for numbers, and a generation step that can say "I don't know".
Measure each fix on a golden set.
*Project example:* "'Weather in Riyadh' scored 0.40, about the same as the correct answer to a
hard question, so a similarity threshold couldn't detect missing answers."

---

## 04 — Generation: grounded answers, citations, "I don't know"

**Q1. How do you write a grounded RAG prompt?**
Static system prompt (cacheable): role, "answer only from the sources", what to do when they don't
contain the answer, partial answers and false premises, scope, injection rules, citation format,
style. Variable user message: the context (market), sources in delimiters with short labels and
metadata, best first, and the question last. Use structured output so status and citations are
machine-readable. Then iterate against a probe set; don't trust the first draft.
*Project example:* "A naive prompt passed 8 of 15 probes and the grounded one 15 of 15; the
failures fixed were mostly product issues, like telling customers 'the provided sources don't say'."

**Q2. How do you implement and verify citations?**
Label sources (S1…Sk), have the model return the labels it used (a structured list, or inline
markers), map them back to chunk IDs in code. Verify deterministically: cited labels must exist, an
answered response must cite something, and numbers in the answer must appear in the cited sources.
Then evaluate citation *support* with an LLM judge or NLI model: does the cited text actually entail
the claim?
*Project example:* "The number check caught nano inventing a USSD code that existed in no source."

**Q3. How do you make a RAG system say "I don't know"?**
Retrieval can't do it: top-k always returns results, and similarity scores overlap between unanswerable
and hard questions. So: an explicit status field in structured output (answered / cannot_answer),
rules telling the model to use it, escalation to a human or support channel, and evaluation on
unanswerable questions in the golden set. Optionally a separate answerability judge, if evals show
it's needed. Handle partial answers and false premises explicitly.

**Q4. Kinds of hallucination in RAG, and how to catch them?**
Fabrication (facts in no source), wrong source (another market or tenant), outdated source,
unfaithful synthesis (distorting what the source says), and out-of-scope answers. Reduce them with
filters (market, freshness), grounding rules and a cannot-answer path. Catch them with deterministic
checks (citations, numbers), LLM-as-judge faithfulness scoring, and monitoring in production.
*Project example:* "Quoting last year's 35 SAR price is prevented by a freshness filter, not by the
prompt."

**Q5. Prompt injection, direct vs indirect; defences?**
Direct: the user's own message tries to override the instructions. Indirect: instructions hidden in
retrieved content (web pages, emails, uploaded files), which is more dangerous because nobody sees
it. Defence in depth: treat inputs as data with clear delimiters; never let the model grant things
directly (discounts and refunds go through tools with authorization); validate outputs in code;
least-privilege tools; red-team cases in the eval set.

**Q6. How many chunks do you send, and in what order?**
k trades recall against cost, latency and distraction; typical values are 3–10, tuned with evals
(recall@k for retrieval, answer quality for generation). Put the most relevant chunk first and the
question last, because models attend most to the start and end of the prompt ("lost in the middle").
Deduplicate near-identical chunks, and use a reranker (Lesson 08) to retrieve wide and send narrow.

**Q7. How do you iterate on a prompt without breaking what works?**
Keep a fixed probe or golden set with explicit expectations; change one thing at a time; run more
than once, because outputs are nondeterministic; version prompts in git alongside their scores; read
the failing outputs, not just the score. Check the test as well as the model: eval sets have bugs.
*Project example:* "One 'unanswerable' probe was actually answered by an FAQ, so the model was right
and my expectation was wrong."

**Q8. How do you keep one market's (or tenant's) answers from reaching another?**
Never infer it from the question: take the market or tenant from the authenticated account and apply
it as a hard metadata pre-filter in retrieval, so the other market's documents never reach the
prompt. Add a prompt rule as a second layer, and test it with probes asking about policies that exist
only for the other market. For strict multi-tenancy, use separate collections or namespaces per
tenant.
*Project example:* "A UAE customer asking about porting gets 'I don't have that information'
instead of the KSA porting rules, because those chunks are filtered out before generation."

---

## 04a — FastAPI

**Q1. Walk through an HTTP request; which status codes would an LLM API return?**
A request is a method, a path, headers and an optional body; the response is a status code, headers
and a body, usually JSON. 2xx means success, 4xx a client error (don't retry unchanged), 5xx a server
error (a retry may help). For an LLM endpoint: 200 for an answer, including "I don't know"; 422 for
an invalid request; 401/403 for auth; 429 when *our* rate limit is hit; 502 when the LLM provider
errors; 503 when not ready or the provider is rate-limiting us (with Retry-After); 504 on an upstream
timeout; 500 only for our own bugs.

**Q2. Why FastAPI over Flask or Django?**
Type hints and Pydantic models drive request parsing, validation, response serialization and the
OpenAPI docs, so you write the contract once. It supports async natively and has dependency
injection built in, and it's the standard for ML and LLM services. Flask is minimal, with validation
and docs as add-ons; Django is full-stack (ORM, admin), which is great for web apps and heavy for a
model-serving microservice.

**Q3. How does FastAPI use Pydantic?**
Request body models are validated before the endpoint runs, so invalid input returns a detailed
422 and never reaches your code. `response_model` validates and *filters* the output, which guards
against leaking internal fields. The models generate the OpenAPI schema and interactive docs.
*Project example:* "AskRequest strips whitespace, caps questions at 500 characters (cost and abuse
control) and forbids unknown fields; AskResponse is a public subset of the internal RAGResponse, with
no costs or retrieval scores."

**Q4. `async def` vs `def`?**
`async def` endpoints run on the event loop: great for non-blocking I/O with `await`, but any blocking
call inside one (a sync HTTP client, `time.sleep`, heavy CPU) freezes every request on that worker.
Plain `def` endpoints run in a thread pool, so blocking calls are safe. Match the endpoint to the
code inside it. LLM calls are network-bound, so a fully async stack (AsyncOpenAI) gives the most
concurrency per process.
*Project example:* "Our pipeline uses the sync OpenAI client, so the endpoints are plain def."

**Q5. Dependency injection in FastAPI, and testing?**
An endpoint declares what it needs (`Depends(get_pipeline)`) and FastAPI provides it, which
centralizes setup (pipelines, database sessions, auth) and enables reuse. For tests,
`app.dependency_overrides` swaps the real dependency for a fake, so tests are fast, free,
deterministic and need no API key. Plus a few `live`-marked tests against the real thing.

**Q6. Error handling for an endpoint that calls an LLM?**
Set client timeouts and retries with backoff for transient errors (the SDK does this). Map failures to
honest codes in central exception handlers: timeout → 504, provider rate limit → 503 with
Retry-After, other provider errors → 502, invalid structured output → 502. Never leak stack traces or
keys; log details with a request ID and return a short message. Consider fallbacks: a cheaper model,
a cached answer, or a hand-off to a human.

**Q7. Load a model or index once, serve many requests and cores?**
Build it once per process, at startup (lifespan) or on first use (a cached dependency), and share it
read-only across requests. One Python process uses one core, so run several workers (uvicorn
`--workers`, or more containers behind a load balancer); each worker holds its own copy, so watch
memory. For big local models, use a dedicated model server (vLLM, Triton) and keep the API thin.
*Project example:* "The pipeline is built once via an lru_cache dependency; if the index is missing,
health returns 503 and the next request retries."

**Q8. Streaming an LLM's answer token by token?**
Users perceive latency as time to first token, so stream. Use `stream=True` on the LLM call and
return a `StreamingResponse` with Server-Sent Events (`text/event-stream`), sending each delta as an
event; the client renders as it arrives. Trade-offs: you can't validate the whole answer before
sending it (so checks such as citation verification run at the end, or you buffer sentences), errors
mid-stream need an in-band error event, and proxies must not buffer.

**Q9. What is an API, and why serve a model behind a web API?**
An API is a contract for how one program asks another for something: the operations, the input
shape, and the output shape including errors. A web API carries that contract over HTTP and JSON,
usually described by an OpenAPI spec. Serving a model behind one means any language or device can
call it, secrets and data (API keys, the index, model weights) stay on the server, you can improve
the internals without redeploying clients, and auth, rate limits, logging and cost tracking live in
one place, and it scales independently. The costs are a network hop and the duty to keep the
contract stable for clients you don't control.
*Project example:* "The OpenAI SDK we call is itself a wrapper over HTTP; our FastAPI service
flips the role and exposes POST /ask, so a mobile app or a WhatsApp bot can use the pipeline
without the OpenAI key or the Chroma index ever leaving our server."

**Q10. What makes an API RESTful? Is `POST /ask` RESTful?**
REST is an architectural style (Fielding, 2000): resources identified by URLs (nouns, plural
collections), a uniform interface of HTTP methods acting on them, JSON representations, meaningful
status codes (201 + Location on create, 204 on delete, 404), statelessness (every request carries
everything needed and shared state lives in a database, so any instance can serve any request), and
cacheable safe reads. Strict REST also requires HATEOAS, which few APIs implement. `POST /ask` is
RPC style: it triggers a computation rather than acting on a stored resource, as OpenAI's
`POST /v1/responses` does. POST fits because the input is structured and potentially long, and the
call is neither cheap nor cacheable. Once conversations exist, I'd remodel it as
`POST /conversations/{id}/messages`, with the history stored server side, and add an
Idempotency-Key for safe retries.

## 04b — Docker

**Q1. Image vs container? Container vs VM?**
An image is a read-only template: stacked filesystem layers plus metadata such as the start
command. A container is a running instance of it with a thin writable layer, so one image can run
as many containers, and a container's writes vanish when it's removed. A VM virtualizes hardware and
runs its own kernel (strong isolation, GBs, slow to boot). A container is just an isolated Linux
process sharing the host kernel, using namespaces for what it can see and cgroups for what it can
use. It's light and starts in under a second, with weaker isolation. On a Mac, Docker runs a small
Linux VM underneath.

**Q2. Why does instruction order matter?**
Each RUN/COPY creates a cached layer, and a change invalidates that layer and every layer after it.
So put rarely changing steps first: copy only the dependency manifests (`pyproject.toml` +
`uv.lock`, or `requirements.txt`), install dependencies, and only then copy the source. A code edit
then reuses the cached dependency layer, and rebuilds take seconds instead of minutes.
*Project example:* "`uv sync --locked --no-dev --no-install-project` runs before `COPY src`; then a
second `uv sync --no-editable` installs only our package."

**Q3. What is a multi-stage build?**
Several FROM stages in one Dockerfile; only the last becomes the image, and it copies just the
artifacts it needs from earlier stages. Build tools, compilers, the package manager and source stay
in the builder. The result is smaller (faster pulls and cold starts) and has less attack surface.
*Project example:* "The builder has uv and installs into a venv; the runtime stage copies only
`.venv` and the index, with no src/ and no uv."

**Q4. How do you give a container secrets?**
At run time, from the environment: `--env-file` or `-e` locally; a secret manager in production
(AWS/GCP Secret Manager, Vault, Kubernetes Secrets) injecting env vars or mounted files. Never COPY
a `.env` and never put keys in ENV or ARG: every layer and the image history are readable by anyone
who can pull the image, and deleting a file in a later layer doesn't remove it from the earlier one.
Add `.dockerignore` so secrets never even enter the build context. For build-time secrets, use
BuildKit `--mount=type=secret`. The same image then runs in every environment with different config.

**Q5. The container runs but `curl localhost:8000` fails. What do you check?**
1. Is the app bound to `0.0.0.0` rather than `127.0.0.1`? Inside a container, localhost is the
   container itself.
2. Is the port published (`-p 8000:8000`, host:container)? EXPOSE alone publishes nothing.
3. Is it still running? `docker ps`, then `docker logs` for a crash (a missing env var, a missing
   index).
4. Does the health status say healthy, or is it still starting?
5. Is something else already using the host port?

**Q6. Why non-root, and why exec-form CMD?**
Containers run as root by default. If the app is exploited, root makes container escapes and damage
far easier, and hardened platforms reject root containers, so create a user and `USER` it, owning
only the folders the app writes to. Exec form (`CMD ["uvicorn", …]`) makes the server PID 1, so it
receives SIGTERM from `docker stop` or Kubernetes and shuts down gracefully, finishing in-flight
requests. Shell form wraps it in `sh`, which doesn't forward signals, so it gets SIGKILLed after the
grace period.

**Q7. Which base image for a Python ML service?**
`python:3.x-slim`, with the version pinned (a digest in production): small Debian with glibc, so
prebuilt manylinux wheels for numpy, onnxruntime, torch and the like install directly. Alpine uses
musl, so those wheels often don't apply: packages compile from source, builds get slow or fail, and
the size advantage disappears. The full image is ~1 GB of tools you don't need. Distroless or
Chainguard is more secure but has no shell for debugging. Use CUDA images only when you need GPUs.

**Q8. Where should state live when containers are disposable?**
Not in the container's writable layer. Logs go to stdout/stderr, collected by the platform.
Durable data goes to a database, object storage (S3) or a named volume. Bind mounts are for
development. Reference data such as a small index can be baked into the image, so code and data
are versioned and rolled back together. A large or frequently updated index belongs in a separate
vector DB service shared by all replicas. Keeping containers stateless is what lets you scale,
restart and replace them freely.
*Project example:* "Our 5 MB Chroma index is baked into the image; logs are bind-mounted only in
development."

## 05 — Retrieval evals

**Q1. Why evaluate retrieval separately from generation?**
Retrieval sets the ceiling: if the right chunk isn't in the context, no prompt can produce a
correct answer. Separate scores say where to work: high hit rate with low faithfulness means fix the
prompt, not the search. Retrieval evals need no LLM, so they're cheap and deterministic enough to run
on every commit.
*Project example:* "Hit@5 is 0.94, so most wrong answers will be generation problems, apart from a
few known slices like Arabic queries."

**Q2. How do you build a golden set with no users yet?**
Combine sources:
- hand-written questions for the known risks and traps (near-duplicate products, regional variants,
  outdated versions, facts buried in long documents);
- real failures found by break-testing;
- LLM-generated questions from sampled chunks, always human-reviewed.

Add unanswerable questions so refusals get tested. Categorize everything so you can slice the
metrics, label relevance carefully, version the set in git, and replace synthetic questions with
real logs as soon as you have users.
*Project example:* "77 questions: 49 hand-written for traps T1–T8, 9 break-test seeds, and 19
synthetic kept from 33 after review."

**Q3. recall@k, precision@k, MRR, nDCG: which do you report for RAG?**
- **Hit@k:** at least one relevant chunk in the top k.
- **Recall@k:** the share of all relevant chunks found.
- **Precision@k:** the share of the top k that is relevant.
- **MRR:** 1 / rank of the first relevant chunk, averaged over questions.
- **nDCG:** graded gains discounted by log2(rank + 1), normalized by the ideal order.

For RAG, report hit@k and recall@k at the k the LLM actually sees (can it answer? does it get
everything?), plus MRR or nDCG for ordering, which matters for lost-in-the-middle and for rerankers
that cut k. Precision is capped by (relevant chunks / k), so it's mostly noise unless context cost
or distraction matters.
*Project example:* "Our multi-doc slice has hit 1.00 but recall 0.68: hit alone would hide
incomplete context."

**Q4. Why do synthetic questions overstate quality?**
The LLM writes the question while looking at the chunk, so it reuses the chunk's words, and
embedding search rewards that overlap. It also produces ambiguous questions that only make sense
with the passage in view, and its distribution follows the corpus rather than real users.
Mitigations:
- ask for paraphrases;
- validate the evidence spans automatically;
- review every candidate;
- write the hard cases by hand;
- move to real queries.

*Project example:* "Synthetic questions scored 1.00 hit@5 while real break-test failures scored
0.67; review dropped 11 of 33 as ambiguous."

**Q5. Recall@5 went from 0.86 to 0.88. Is that real?**
Probably not, by itself. With about 70 questions, the bootstrap 95% interval is roughly ±0.06, and
0.02 is one or two questions. Do a paired comparison on the same questions: which ones changed and
why. Use more questions, or a significance test (paired bootstrap, or McNemar for hit rate), before
claiming an improvement, and check that no category regressed.

**Q6. How do you label relevance so labels survive re-chunking?**
Don't store chunk IDs; they change with any chunking change. Label by content: "a chunk of document
D containing text T" (a fact or number), resolved against the current corpus at eval time. Grade it
(answers / partial). Test the labels: every label must match at least one current, in-market chunk.
*Project example:* "`{doc: roaming-policy-uae, has: '20 AED for 1 GB'}`, matched ignoring case and
markdown."

**Q7. A question fails retrieval. What do you check first?**
1. Look at what *was* retrieved. Is the label incomplete, with the same fact elsewhere worded
   differently?
2. Is the question itself ambiguous or wrong?
3. Did a filter exclude the document by design?
4. Is the right chunk just below the cut (the hit curve at 10 vs 5)?

Only then is it a retriever problem, and the kind of miss points to the fix: vocabulary mismatch
suggests hybrid search or query rewriting; just below the cut suggests a reranker; another language
suggests translation or a multilingual model.
*Project example:* "1 of our 5 first-run misses was a labelling error: the FAQ said '150 or more'
where the policy said '150 SAR or 150 AED'."

**Q8. Dev set vs held-out test set?**
You tune prompts, chunking and retrieval against the dev set; repeated tuning overfits to its
specific questions (Goodhart's law). A held-out test set is run rarely, only for final numbers, so
it gives an unbiased estimate of quality on unseen questions. Keep them from the same distribution,
version both, and never tune on the test set; if you do, refresh it.
