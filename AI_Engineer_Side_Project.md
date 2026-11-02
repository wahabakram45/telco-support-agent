# AI Engineer Side Project — Telecom Support RAG Agent

*Parked idea (28 Oct 2026). Come back to it after career-ops is set up.*

## Why this project

The CV shows strong production ML (2M+ subscriber models, Databricks, Kubernetes) and one offline LLM pipeline.
AI Engineer job descriptions keep asking for things it doesn't show yet: **RAG, agents/tool calling, LLM evals,
and serving LLM apps in production**. One end-to-end project closes that gap and gives an interview story
that connects to the telecom domain.

## The idea

A support assistant for a (fictional or public-data) mobile operator that answers questions about bundles, plans,
roaming and billing, and can call tools — e.g. "which bundle fits someone who uses 2 GB data and 300 minutes a month?"

Use public plan/FAQ pages from operators, or write a synthetic catalog. **Never use Beyond ONE / Virgin / FRiENDi
internal data.**

## Scope (what makes it "AI Engineer" and not a demo)

| Area | What to build | Skill it proves |
|---|---|---|
| Ingestion | Chunk plan docs + FAQs; metadata (plan, price, validity, market) | Data prep for LLMs |
| Retrieval | Hybrid search: BM25 + embeddings (pgvector / Qdrant / Chroma), then a reranker | RAG done properly |
| Agent | Tool calling: `search_plans`, `recommend_bundle(usage)`, `compare_plans` | Agents / function calling |
| **Evals (the differentiator)** | Golden Q&A set (50–100), retrieval recall@k, answer faithfulness via LLM-as-judge, tool-call accuracy; runs in CI as a regression gate | LLM evaluation |
| Serving | FastAPI + Docker, streaming responses | Production APIs |
| Observability | Tracing (Langfuse / OpenTelemetry), latency, tokens and cost per query | LLMOps |
| Guardrails | Refuse out-of-scope questions, cite sources, never invent prices | Safety / reliability |

## Suggested milestones

1. **Week 1:** Naive RAG end to end (chunk → embed → retrieve → answer) behind FastAPI.
2. **Week 2:** Build the eval set and harness *before* optimizing. Measure the baseline.
3. **Week 3:** Hybrid search + reranker; show the eval numbers improving.
4. **Week 4:** Add tools/agent loop, tracing and cost tracking; Dockerize; write the README with results.

## Good engineering practices to show

- Typed Python (pydantic), tests, `ruff`, pre-commit
- Config via env files, no secrets in git
- Clear README: architecture diagram, eval results table, cost per 1k queries, trade-offs you made

## What goes on the CV when done

One project entry with 2–3 bullets and real numbers from your own evals — e.g. retrieval recall@5 before/after
reranking, faithfulness score, p95 latency, cost per query. Link the GitHub repo.

## Reference for eval design

career-ops itself has a golden-set eval harness (`eval-golden.mjs`, `evals/golden/`) that gates cheaper models
against reference verdicts — a useful pattern to read before designing your own.
