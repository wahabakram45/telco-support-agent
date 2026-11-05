# Lesson 04a — FastAPI: putting the pipeline behind an HTTP API

> **Goal:** understand HTTP and REST well enough to design an API, and learn FastAPI's core:
> routes, Pydantic request/response models, validation, docs, async vs sync, dependency injection,
> error handling and testing. Wrap the RAG pipeline in `POST /ask` and `GET /health`.
> **Time:** ~3 hours (1.5 h reading, 45 min building, 15 min exploring, 30 min quiz).

Until now the pipeline runs as a script on your laptop. A web app, a mobile app or a WhatsApp bot
needs to call it over the network. That's what an API is for.

---

## Part A — Concepts

### 1. HTTP in 5 minutes

A **client** (browser, app, `curl`) sends a **request** to a **server**, and gets back a **response**.

```
POST /ask HTTP/1.1                         ← method + path
Host: api.telcox.example                   ← headers: metadata about the request
Content-Type: application/json
X-Request-ID: 7f3a9c

{"question": "How much is the GCC day pass?", "market": "KSA"}   ← body (JSON)
```
```
HTTP/1.1 200 OK                            ← status code
Content-Type: application/json
X-Request-ID: 7f3a9c

{"status": "answered", "answer": "The GCC Day Pass costs 25 SAR…", "citations": […]}
```

**Methods** say what kind of action you want:

| Method | Meaning | Safe (no changes)? | Idempotent (repeat = same effect)? |
|---|---|---|---|
| GET | Read something | ✅ | ✅ |
| POST | Create or process something | ❌ | ❌ |
| PUT | Replace something | ❌ | ✅ |
| PATCH | Change part of something | ❌ | ❌ |
| DELETE | Remove something | ❌ | ✅ |

**Status codes** tell the client what happened. The first digit is the category: **2xx** success,
**4xx** the *client* did something wrong (don't retry unchanged), **5xx** the *server* failed (a retry
may work). The ones our API uses:

| Code | Meaning | When we return it |
|---|---|---|
| 200 OK | Success | An answer, including "cannot_answer" (the request worked; the answer is "I don't know") |
| 422 Unprocessable Content | The body is well-formed JSON but invalid | Missing market, empty question… (FastAPI does this automatically) |
| 404 Not Found | No such path | Calling `/asks` by mistake |
| 502 Bad Gateway | A service *we depend on* failed | OpenAI error, or no structured answer |
| 503 Service Unavailable | We can't serve right now | Index not built; OpenAI rate-limited us (with `Retry-After`) |
| 504 Gateway Timeout | A service we depend on was too slow | OpenAI timed out |
| 500 Internal Server Error | Our own bug | Anything unhandled |

### 2. REST, and where `/ask` fits

**REST** is a style: URLs name **resources** (nouns: `/plans`, `/plans/flex-smart-ksa`), and the
method says what to do with them (`GET /plans/flex-smart-ksa`). Requests are **stateless**: each one
carries everything needed, and the server keeps no session between them.

`POST /ask` is really **RPC style** ("do this action"), not a pure resource. That's normal for ML
and LLM services: OpenAI's own API is `POST /v1/responses`, `POST /v1/embeddings`. Use POST because
the input can be long and structured, and the call isn't a simple read. In an interview, knowing
*why* it isn't pure REST matters more than forcing it to be.

### 3. Why FastAPI

| Framework | Style | Notes |
|---|---|---|
| Flask | Minimal, sync | Huge ecosystem; validation and docs are add-ons |
| Django + DRF | Full-stack, batteries included | ORM, admin, auth; heavy for an ML microservice |
| **FastAPI** | Type hints drive everything | Validation, serialization and OpenAPI docs come from your Pydantic models and type hints; async support; the de facto standard for ML and LLM services |
| Litestar | Similar idea to FastAPI | Smaller community |

The core idea: **you write type hints and Pydantic models once**, and FastAPI uses them to parse and
validate requests, serialize responses, and generate interactive documentation.

### 4. Anatomy of an endpoint

```python
@app.post("/ask", response_model=AskResponse)             # method, path, response contract
def ask(request: AskRequest, pipeline: PipelineDep):      # body model, injected dependency
    return AskResponse.from_rag(pipeline.ask(request.question, request.market))
```
- **Where parameters come from:** a Pydantic model parameter → the JSON **body**; a name in the path
  (`/plans/{plan_id}`) → a **path parameter**; any other simple type (`k: int = 5`) → a **query
  parameter** (`/ask?k=3`).
- **Validation:** the body is checked against `AskRequest` *before* your function runs. Invalid
  input never reaches your code; the client gets a 422 explaining exactly what's wrong.
- **`response_model`:** the response is validated and **filtered** to exactly the declared fields.
  That's a safety net against leaking internal data.
- **Docs for free:** `/docs` (interactive Swagger UI, where you can try requests in the browser),
  `/redoc`, and the raw `/openapi.json` that other teams can generate client code from.

**Internal vs public models.** `RAGResponse` holds costs, retrieval scores and raw checks: useful to
us, none of the client's business, and things we want freedom to change. `AskResponse` is the
**public contract**: a deliberate subset that changes only on purpose. Keep the two separate.

### 5. `async def` vs `def`, the classic interview question

FastAPI runs on an **event loop** (one thread juggling many requests).
- **`async def` endpoints** run *on* the event loop. Use them only when everything inside is async
  (`await`). **A blocking call inside `async def` (a sync HTTP client, `time.sleep`, heavy CPU)
  freezes the entire server** for every user until it finishes.
- **`def` endpoints** run in a **thread pool**, so blocking calls are fine: other requests keep
  being served.

Our pipeline uses the **sync** OpenAI client and Chroma, so our endpoints are plain **`def`**.
Going fully async later (`AsyncOpenAI`, `await`) would let one process handle many more
concurrent LLM calls, since they spend most of their time waiting on the network.

### 6. Dependency injection

```python
@lru_cache
def get_pipeline() -> RAGPipeline: ...
PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
```
An endpoint *declares* what it needs; FastAPI *provides* it. Why bother:
- **Load expensive things once.** The pipeline opens Chroma and the embedding cache; doing that per
  request would be slow. `lru_cache` builds it on first use and shares it. (The alternative is
  FastAPI's **lifespan** hook, which builds it at startup, so the first request isn't slow and a
  broken index fails at boot. Both are common.)
- **Testability.** `app.dependency_overrides[get_pipeline] = lambda: FakePipeline()` swaps in a
  fake: tests run in milliseconds, free, with no API key.
- **Reuse.** The same dependency can do auth, database sessions, rate limiting…

### 7. Errors

- **Map failures to honest status codes** (table in §1): it tells clients whether a retry can help.
- **Exception handlers** (`@app.exception_handler(openai.APITimeoutError)`) turn exceptions into
  clean JSON responses in one place, instead of `try/except` in every route. Handlers are matched
  by class hierarchy, so `APITimeoutError` gets its own 504, and other `APIError`s fall back to 502.
- **Never leak internals** (stack traces, file paths, keys) in responses. Log the details server
  side, with the **request ID**, and return a short message. A client reporting "request 7f3a9c
  failed" lets you find the exact log line.

### 8. Health checks

`GET /health` answers "can this instance serve traffic?" It checks that the index is loaded, and
deliberately **makes no LLM call**: health checks run every few seconds, so they must be fast and
free. Orchestrators distinguish **liveness** (is the process alive? restart it if not) from
**readiness** (can it serve? stop sending traffic if not). Ours is a readiness check.

### 9. Running it

- **Uvicorn** is the **ASGI server**: it handles the network and calls your FastAPI app.
  FastAPI itself doesn't listen on a port.
- Development: `fastapi dev` (auto-reload on code changes). Production: `fastapi run` or
  `uvicorn … --workers 4`: several processes, because one Python process uses one CPU core. Usually
  behind a reverse proxy or load balancer. Docker in Lesson 04b.

### 10. Our decisions

- `POST /ask` (RPC style) and `GET /health`; the market in the body for now. (In production it
  comes from the authenticated user, never from a field the client can set freely: Phase 7.)
- A separate public `AskResponse`, with no costs or scores.
- Sync `def` endpoints, matching our sync pipeline.
- `lru_cache` dependency, overridden with fakes in tests.
- 502/503/504 for upstream failures, a request ID on every response, and no internals in errors.

---

## Part B — Build

I wrote: the response and health models, the dependency, the middleware, the exception handlers,
`/health`, and `tests/test_api.py`. **You write the two pieces at the heart of FastAPI.**

### Task 1 — Validation rules for `AskRequest` (~15 min)
In `src/telco_support_agent/api.py`. The tests want:
- whitespace stripped from the question (`"  How much?  "` → `"How much?"`);
- the question at least 1 character (after stripping) and at most 500;
- unknown fields rejected (`"customer_id": "42"` → 422);
- the market only `KSA` or `UAE` (already done by the `Market` type).

Hints: `ConfigDict(extra=..., str_strip_whitespace=...)`, and `Field(min_length=..., max_length=...)`
(import `Field` from pydantic). Why cap the length? Every character costs tokens, and a 50,000-
character "question" is either a mistake or an attack.

### Task 2 — The `POST /ask` route (~15 min)
At the bottom of `api.py`: a sync route at `/ask` that takes an `AskRequest` body and the
`PipelineDep`, calls `pipeline.ask(...)`, and returns `AskResponse.from_rag(...)`. Declare
`response_model=AskResponse`. It's three lines; the point is knowing why each piece is there.

`uv run pytest tests/test_api.py -v` until it's green (16 tests).

### Task 3 — Run it and explore (~15 min)
```bash
uv run fastapi dev src/telco_support_agent/api.py
```
1. Open **http://127.0.0.1:8000/docs**. Expand `POST /ask`, click "Try it out", and send a question.
   Look at the schema FastAPI generated from your models.
2. Send an invalid body (market `"PK"`, empty question) and read the 422.
3. From a second terminal:
   ```bash
   curl -s localhost:8000/health
   curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
        -d '{"question": "How much is the GCC day pass?", "market": "UAE"}'
   ```
4. Watch the server terminal: one log line per request, with the request ID.

When you're done, commit and say "lesson 04a done". I'll clean up, then quiz you.

---

## Part C — Interview drill

Short-format quiz on "review lesson 04a". Full model answers:
[interview-bank §04a](../interview-bank.md#04a--fastapi).

1. Walk through an HTTP request and response. Which status codes would an LLM API return, and when?
2. Why FastAPI over Flask or Django?
3. How does FastAPI use Pydantic?
4. `async def` vs `def`: what happens with a blocking call inside `async def`?
5. What is dependency injection in FastAPI, and how does it help testing?
6. How do you handle errors in an endpoint that calls an LLM?
7. How do you load a model or index once and serve many requests (and many CPU cores)?
8. How would you stream an LLM's answer token by token to the client?
