# Roadmap — Telecom Support RAG Agent

Plan: ~8 weeks at 10–15 h/week (FastAPI + Docker lessons included). Each phase = lesson(s) in `learn/` → you build → we measure → interview drill.
Original spec: [AI_Engineer_Side_Project.md](AI_Engineer_Side_Project.md). Gaps added from your evaluated JDs are marked ➕.

**Rule of the project:** no change ships without an eval number that moved.
**Quizzes:** short format only (multiple choice, one-liners, keywords). Full model answers live in `interview-bank.md` for reading or speaking practice.
**Coding split:** Claude writes glue (config, boilerplate, tests). You write the core parts, marked `# TODO(you)`.

---

## Tooling to install along the way
- Phase 0: `uv` (Python), `git`, `pre-commit` (installed via uv)
- Phase 2: **OrbStack** (`brew install orbstack`). It runs Docker on Mac and is much lighter than Docker Desktop,
  which matters on 8 GB of RAM. The `docker` commands you learn are identical.

## Phase 0 — Python toolkit & LLM API fundamentals  (week 1, ~8 h)
- [x] Lesson [00a — Python engineering toolkit](learn/00a-python-toolkit.md) ➕: uv, type hints, **Pydantic**
      (models, validation, Field, JSON Schema, settings), **pytest** (fixtures, parametrize, fakes), **ruff** (lint + format), **pre-commit**
- [x] Build: add validation rules to `UsageProfile`, install pre-commit and watch it block a bad commit
- [x] Drill: interview-bank §00a (8/8)
- [x] Lesson [00b — LLM API fundamentals](learn/00b-llm-api-fundamentals.md): tokens, context windows, sampling, cost, ➕ structured outputs
- [x] Build: `llm.py` wrapper that logs tokens and cost for every call, plus structured extraction of a usage profile
- [x] Drill: interview-bank §00b (8/8)

Pydantic comes back throughout the project: request and response models in FastAPI (04a), tool schemas
(10), eval records (05). Ruff and pre-commit run on every commit from now on, and later in CI (Phase 3).

## Phase 1 — The data  (week 1, ~5–6 h)
- [x] Lesson [01 — Data for RAG](learn/01-data-for-rag.md): structured vs unstructured, metadata, real vs synthetic data, traps, freshness, PII
- [x] Build: `Plan` model, a catalog of ≥30 current plans with traps T1–T5, number checker, generation prompt → ~55–65 documents
- [x] Drill: interview-bank §01 (7/8; missed: embeddings are not anonymisation)

## Phase 2 — Naive RAG end to end + FastAPI & Docker basics  (weeks 2–3)
- [x] Lesson [02 — Chunking](learn/02-chunking.md): fixed, recursive, structure-aware, semantic; size, overlap, contextual headers
- [x] Build: `chunk_by_headings()` + strategy comparison
- [x] Drill: interview-bank §02 — 7/8 (missed: fixed-500 = one chunk per doc → diluted; proxy metric can't see it)
- [x] Lesson [03 — Embeddings & vector search](learn/03-embeddings-vector-search.md): what embeddings are, cosine similarity, ANN indexes (HNSW), vector DB options
- [x] Build: `cosine_top_k()` + index the corpus + probe what embeddings miss
- [x] Drill: interview-bank §03 — 6/8 (missed: unit vectors → cosine, dot, L2 all rank the same; ef_search is the query-time knob)
- [x] Lesson [04 — Generation](learn/04-generation.md): grounding prompts, citations, "I don't know", prompt injection, Chroma
- [x] Build: ingest → chunk → embed → Chroma → retrieve → answer with citations; write the grounded system prompt (≥14/15 probes)
- [x] Drill: interview-bank §04 — 8/8
- [x] Lesson [04a — ➕ FastAPI fundamentals](learn/04a-fastapi.md): HTTP & REST basics, routes, Pydantic request/response models,
      automatic validation & OpenAPI docs, status codes & error handling, dependency injection, testing with `TestClient`
- [x] Build: `/ask` and `/health` endpoints around the RAG pipeline, with tests
- [x] Drill: interview-bank §04a — 5/8 (missed: GET must be safe → close via PATCH; tests use dependency_overrides not the live marker; /health makes no LLM call: runs every few seconds, must be fast and free)
- [ ] Lesson 04b — ➕ Docker fundamentals: images vs containers, Dockerfile line by line, layers & caching,
      ports, env vars & secrets, volumes, `.dockerignore`
- [ ] Build: containerize the API; `docker run` it locally

## Phase 3 — Evals FIRST  (week 4) ⭐ the differentiator
- [ ] Lesson 05 — Retrieval metrics: recall@k, precision@k, MRR, nDCG; building a golden set
- [ ] Lesson 06 — Generation evals: faithfulness, answer relevance, correctness; LLM-as-judge and calibrating the judge against human labels
- [ ] Build: 80-question golden set, eval harness, baseline numbers, CI regression gate
- [ ] Record baseline in the results table below

## Phase 4 — Better retrieval  (week 5)
- [ ] Lesson 07 — BM25 & hybrid search: sparse vs dense, Reciprocal Rank Fusion
- [ ] Lesson 08 — Rerankers: bi-encoder vs cross-encoder, latency trade-off
- [ ] Lesson 09 — ➕ Query understanding: query rewriting, metadata filters, multi-turn follow-ups ("and what about roaming?")
- [ ] Build each one and show its eval delta

## Phase 5 — Agents & tool calling  (week 6)
- [ ] Lesson 10 — Function calling from scratch: the agent loop, tool schemas, stop conditions, error handling
- [ ] Lesson 11 — Agentic patterns: ReAct, planning, routing, memory, human-in-the-loop, when NOT to use an agent
- [ ] Build: `search_plans`, `recommend_bundle(usage)`, `compare_plans`; tool-call accuracy evals

## Phase 6 — Production  (week 7)
- [ ] Lesson 12 — ➕ FastAPI in production: async vs sync (the event loop, why blocking calls hurt),
      streaming responses (SSE), lifespan startup (load models once), middleware (request IDs, timing), workers & concurrency
- [ ] Lesson 12a — ➕ Docker in production: multi-stage builds & small images, non-root user, healthchecks,
      Docker Compose (API + Langfuse + Postgres), building the image in CI
- [ ] Lesson 13 — Observability: tracing with Langfuse, latency p50/p95, ➕ prompt caching and cost per 1k queries
- [ ] Lesson 14 — Guardrails: out-of-scope refusal, price-hallucination checks, prompt injection
- [ ] Lesson 15 — ➕ Prompt experiments & failure-mode analysis: versioned prompts, A/B on the eval set, an error taxonomy

## Phase 7 — Frameworks & fine-tuning  (week 8)
- [ ] Lesson 16 — ➕ Rebuild the agent in LangGraph: what it gives you, what it hides
- [ ] Lesson 17 — ➕ MCP: expose the tools as an MCP server
- [ ] Lesson 18 — ➕ Fine-tuning an embedding model with HuggingFace on synthetic queries (Colab), measured with recall@k; when to fine-tune vs RAG vs prompting
- [ ] Final: README with architecture diagram and results, CV bullets, full mock interview

---

## Results table (filled in as we go — this becomes the README and your CV numbers)

| Version | Recall@5 | MRR | Faithfulness | Answer correctness | Tool-call acc. | p95 latency | Cost / 1k queries |
|---|---|---|---|---|---|---|---|
| v0 naive RAG | | | | | – | | |
| v1 + hybrid | | | | | – | | |
| v2 + reranker | | | | | – | | |
| v3 + agent | | | | | | | |
| v4 + fine-tuned embeddings | | | | | | | |
