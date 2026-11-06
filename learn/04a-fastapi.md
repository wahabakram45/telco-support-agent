# Lesson 04a — FastAPI: putting the pipeline behind an HTTP API

> **Goal:** understand HTTP and REST well enough to design an API, and learn FastAPI's core:
> routes, Pydantic request/response models, validation, docs, async vs sync, dependency injection,
> error handling and testing. Wrap the RAG pipeline in `POST /ask` and `GET /health`.
> **Time:** ~3 hours (1.5 h reading, 45 min building, 15 min exploring, 30 min quiz).

Until now the pipeline runs as a script on your laptop. A web app, a mobile app or a WhatsApp bot
needs to call it over the network. That's what an API is for.

---

## Part A — Concepts

### 0. What an API really is

**API = Application Programming Interface: a contract for how one program asks another program
for something.** Think of a restaurant menu. It lists what you can order, what you have to tell the
waiter ("medium, no onions") and what comes back. You never enter the kitchen, and the kitchen can
change its stoves without changing the menu.

You've used APIs this whole project, at three levels:

| Kind | Example from our code | The "contract" |
|---|---|---|
| Library API | `np.argsort(scores)`, `Embedder.embed(texts)` | Function names, arguments, return types |
| OS API | `Path.read_text()` → the OS opens the file | System calls |
| **Web API** | `client.responses.create(...)` → OpenAI's servers | HTTP requests and JSON over the network |

The last one is what this lesson is about. The OpenAI Python SDK is only a convenience wrapper. Under
it, every call is a plain HTTP request:

```bash
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": "GCC day pass"}'
```

Now we flip roles. Until now we were the **client** of OpenAI's API. Today we become a **server**
with our own API that other programs call.

**Every API contract defines three things:**
1. **Operations:** what you can ask for (`POST /ask`, `GET /health`).
2. **Input shape:** what you must send (`{"question": str, "market": "KSA" | "UAE"}`).
3. **Output shape and errors:** what comes back on success and on each kind of failure.

FastAPI writes this contract down for you as an **OpenAPI** document, a machine-readable spec that
other teams can generate client code from.

**Why put the pipeline behind a web API instead of letting people `import` it?**
- **Any language, any device.** A Swift iOS app, a Node.js WhatsApp bot and a React website can't
  import Python. All of them can send HTTP.
- **Secrets and data stay on the server.** The OpenAI key and the Chroma index never leave our
  machine. A client only ever sees answers.
- **One place to change.** Improve the prompt or swap the vector DB, and every client benefits
  without shipping a new app. The contract stays the same; the kitchen changes.
- **Control.** Authentication, rate limits, logging, cost tracking: all in one place.
- **Independent scaling.** Run 10 copies of the API on busy days without touching the clients.

The cost: a network hop (milliseconds; small next to a 1–2 s LLM call), and you have to keep the
contract stable, because clients you don't control depend on it.

**API styles.** "Web API" covers several styles:

| Style | Shape | Typical use |
|---|---|---|
| **REST** | Resources at URLs, HTTP methods as verbs: `GET /plans/flex-smart-ksa` | Most public web APIs; CRUD on data |
| **RPC over HTTP** | Actions: `POST /ask`, `POST /v1/embeddings` | ML/LLM services, "do this computation" |
| **gRPC** | RPC with binary Protobuf messages, generated clients, HTTP/2 | Fast service-to-service calls inside a company |
| **GraphQL** | One endpoint; the client writes a query for exactly the fields it wants | Frontends that need flexible, nested data |
| **Streaming (SSE, WebSocket)** | The server pushes data over a long-lived connection | LLM tokens as they're generated, live chat |
| **Webhooks** | Reversed: *the server calls you* when something happens | "Payment succeeded", "ticket closed" |

The main split is **who starts the conversation, and how long it lasts**:
- **Request → response** (the client asks, the server answers once): REST, RPC, gRPC, GraphQL.
- **The server keeps sending** over an open connection: SSE, WebSocket.
- **The server starts it**, later, on its own: webhooks.

#### One task, six ways: the differences side by side
**The task:** the TelcoX app shows customer 42 *their plan's price* and *their open tickets*, and
then tells them when a ticket gets closed.

**REST:** two resources, so two requests. Each returns the *whole* resource.
```
GET /v1/customers/42/plan
→ {"id": "flex-smart-ksa", "name": "Flex Smart", "price": 95, "data_gb": 40, "minutes": 500, …}
GET /v1/customers/42/tickets?status=open
→ [{"id": "T-123", "subject": "Roaming not working", "created": "…", "priority": "…", …}]
```
2 round trips, plus fields the screen doesn't need (`data_gb`, `minutes`…).

**RPC over HTTP:** the backend writes one action for exactly this screen.
```
POST /v1/getHomeScreen   {"customer_id": 42}
→ {"plan_price": 95, "open_tickets": [{"id": "T-123", "subject": "Roaming not working"}]}
```
1 round trip and a perfect fit, but only for this screen. A new screen needs a new action
(`getBillingScreen`, `getUsageScreen`…).

**gRPC:** the same RPC idea, but typed and binary, from generated code.
```python
reply = stub.GetHomeScreen(HomeScreenRequest(customer_id=42))
reply.plan_price        # 95 — a typo like reply.plan_prise fails immediately
```
On the wire it's a few compact binary bytes, unreadable to `curl`. It's fast, but a browser can't
call it directly.

**GraphQL:** one generic endpoint, and the *client* picks the fields.
```graphql
query { customer(id: 42) { plan { price } tickets(status: OPEN) { id subject } } }
→ {"data": {"customer": {"plan": {"price": 95},
                         "tickets": [{"id": "T-123", "subject": "Roaming not working"}]}}}
```
1 round trip, exactly the fields asked for, and no new backend code when the next screen needs
`plan.data_gb` too.

**SSE:** a different shape of problem, used when the result arrives *gradually*. Say the customer
asks the support bot about the ticket:
```
POST /ask/stream   {"question": "Why is roaming not working?", "market": "KSA"}
← data: {"delta": "Roaming needs"}
← data: {"delta": " to be enabled in"}
← data: {"delta": " the app under Settings."}
← data: [DONE]
```
1 request, **many** responses: the first words show after about 300 ms instead of after 2 s.

**Webhook:** the *other side* starts it. Nobody asked; the ticket system announces the close:
```
POST https://telcox-app.example/hooks/tickets      ← sent BY the ticket system TO the app backend
{"event": "ticket.closed", "ticket_id": "T-123", "customer_id": 42}
```
Without it, the app would poll `GET /tickets/T-123` every few seconds until the status changes.

**The same comparison as a table:**

| | REST | RPC/HTTP | gRPC | GraphQL | SSE | Webhook |
|---|---|---|---|---|---|---|
| Who starts | Client | Client | Client | Client | Client, once | **Server** |
| Responses per request | 1 | 1 | 1 (or a stream) | 1 | **Many** | — (it *is* the request) |
| Round trips for our screen | 2 | 1 | 1 | 1 | — | — |
| Who decides the fields | Server | Server | Server (`.proto`) | **Client** | Server | Sender |
| URL names | A thing (`/plans/…`) | An action (`/getHomeScreen`) | A method (`GetHomeScreen`) | One endpoint (`/graphql`) | An action | Your receiving URL |
| Format | JSON | JSON | **Binary Protobuf** | JSON | Text events | JSON |
| Readable with `curl` / browser | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Easy HTTP caching | ✅ (GET) | ❌ | ❌ | ❌ | ❌ | ❌ |
| New screen needs backend change? | Sometimes | **Yes** | **Yes** | Rarely | — | — |

Read the table this way:
- **REST, RPC, gRPC and GraphQL** all answer the *same* kind of question: one request, one answer.
  They differ in what the URL names and who chooses the fields.
- **SSE and webhooks** solve *different* timing problems. SSE handles an answer that arrives in
  pieces; webhooks handle an event that happens later, which nobody asked about.

Below, each style in more depth.

#### REST: "here are my things, act on them"
```
GET /v1/plans/flex-smart-ksa
→ 200 {"id": "flex-smart-ksa", "price": 95, "currency": "SAR", "data_gb": 40}
```
- **How it works:** every thing has a URL; the four HTTP verbs act on it (§2 covers it in depth).
- **Strengths:** predictable; works with any HTTP tool (browser, `curl`); GET responses can be
  cached by browsers and CDNs; huge tooling (OpenAPI, Swagger).
- **Weaknesses:** a screen often needs several calls (plan + customer + usage = 3 requests:
  **under-fetching**), and each response returns every field even if you need one
  (**over-fetching**). Awkward for actions that aren't "create/read/update/delete a thing".
- **Use it for:** public APIs and data you store: plans, tickets, customers.

#### RPC over HTTP: "run this function for me"
```
POST /ask   {"question": "How much is the GCC day pass?", "market": "KSA"}
→ 200 {"status": "answered", "answer": "25 SAR…", "citations": […]}
```
- **How it works:** RPC = **Remote Procedure Call**. The URL names an *action* and the body holds
  the arguments, like calling a Python function that happens to run on another machine. Almost
  always POST with JSON.
- **Strengths:** natural for computations that don't fit "a thing": answer, embed, classify,
  translate. Simple to design.
- **Weaknesses:** no caching; every API invents its own verbs, so it's less predictable than REST.
- **Use it for:** ML and LLM services. That's **us**, and OpenAI (`POST /v1/embeddings`).

#### gRPC: RPC, made fast and strict, for services talking to each other
You first write the contract in a `.proto` file:
```proto
service Support {
  rpc Ask (AskRequest) returns (AskReply);
}
message AskRequest { string question = 1; string market = 2; }
```
A tool then **generates** client and server code in Python, Go, Java… and you call
`stub.Ask(AskRequest(question=…, market="KSA"))` as if it were a local function.
- **How it works:** Google's RPC framework. Messages are **Protobuf**, a compact binary format
  (smaller and faster to parse than JSON), sent over **HTTP/2** (many calls share one connection,
  and both sides can stream).
- **Strengths:** fast; the contract is enforced (a typo is a compile error, not a 422 at runtime);
  built-in streaming; client libraries generated for free.
- **Weaknesses:** browsers can't call it directly (it needs a proxy); binary messages can't be read
  with `curl` or in logs; more setup.
- **Use it for:** internal service-to-service traffic where speed matters. For example, if our API
  called a separate in-house reranker service thousands of times a second. Model servers like
  Triton and many vector DBs (Qdrant, Milvus) offer gRPC next to REST for this reason.

#### GraphQL: "here's exactly the data I want"
One endpoint (`POST /graphql`); the client sends a *query* describing the shape it wants:
```graphql
query {
  customer(id: 42) {
    name
    plan { name price }
    tickets(status: OPEN) { id subject }
  }
}
```
and gets back JSON in exactly that shape, in **one** round trip, with no unused fields.
- **How it works:** the server publishes a typed **schema** of everything that can be asked for;
  resolvers fetch each field. Created by Facebook for its mobile app.
- **Strengths:** solves REST's under/over-fetching; great when many different screens need
  different slices of the same connected data; the frontend team can change what it fetches without
  waiting for the backend to add endpoints.
- **Weaknesses:** harder to cache (everything is a POST to one URL); one careless query can be very
  expensive (deeply nested), so you need query limits; more complex server.
- **Use it for:** a customer-facing app whose home screen shows profile + plan + usage + tickets.
  Not for us: we have one operation, and nothing to "select".

#### Streaming: SSE and WebSocket — "keep the line open"
Normal HTTP is one request, one response, then done. Streaming keeps the connection open so the
server can send many messages.
- **SSE (Server-Sent Events):** the client makes one normal request; the server responds with
  `Content-Type: text/event-stream` and keeps writing events:
  ```
  data: {"delta": "The GCC"}
  data: {"delta": " Day Pass costs"}
  data: {"delta": " 25 SAR."}
  data: [DONE]
  ```
  **One direction** (server → client), plain HTTP, so it works through proxies and is simple. This is
  how ChatGPT and OpenAI's `stream=True` deliver tokens. **For an LLM answer, this is the default
  choice**, because users judge speed by when the first words appear.
- **WebSocket:** the connection is "upgraded" from HTTP into a **two-way** channel; either side can
  send at any time. Needed when the client also sends continuously: live chat with a human agent,
  typing indicators, voice assistants streaming audio both ways (OpenAI's Realtime API).
- **Weaknesses:** long-lived connections are harder to load-balance and scale; you must handle
  disconnects and reconnects; errors can happen halfway through a response.

#### Webhooks: "don't call us, we'll call you"
You give another service a URL of yours; when an event happens, **it** sends an HTTP POST to you:
```
POST https://telcox.example/hooks/payments      ← sent by the payment provider
{"event": "payment.succeeded", "customer_id": 42, "amount": 95, "currency": "SAR"}
```
- **Why:** the alternative is **polling**, asking "is it paid yet?" every 5 seconds: wasteful, and
  slow to notice. A webhook tells you the moment it happens.
- **Things to get right:** verify the sender (a signature header, usually HMAC with a shared
  secret), since anyone can POST to a public URL; reply 2xx quickly and do the work in the
  background; expect **duplicates and retries**, so process each event idempotently (by its ID).
- **Use it for:** events from other systems: payments (Stripe), messages (the WhatsApp Business API
  delivers incoming customer messages to your webhook), CI finished (GitHub).

#### How they fit together in one real product
A real TelcoX support setup would use several at once:

```
Customer on WhatsApp ──(WhatsApp sends a webhook)──► TelcoX bot backend
Bot backend ──(RPC: POST /ask)──► our RAG API ──(RPC: POST /v1/responses)──► OpenAI
Web chat widget ◄──(SSE: tokens as they arrive)── our RAG API
Mobile app home screen ──(GraphQL or REST)──► customer/plan/usage data
RAG API ──(gRPC, internal)──► a reranker service (later)
```

#### Choosing, in one line each
- Exposing **stored data** to many unknown clients → **REST**.
- Exposing a **computation** (a model, a pipeline) → **RPC over HTTP** (what we're building).
- **Internal**, high-volume, service to service → **gRPC**.
- **Many screens**, each needing different slices of connected data → **GraphQL**.
- The server must **push** as it goes → **SSE** (one way) or **WebSocket** (two way).
- **React to events** in someone else's system → **webhooks**.

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

**REST = REpresentational State Transfer.** It isn't a library, a protocol or a standard you
install. It's an **architectural style**, a set of design rules, that Roy Fielding described in his
2000 PhD thesis by studying why the web itself scaled so well. An API that follows the rules is
called **RESTful**.

**The core idea: everything is a *resource* with an address, and a few standard verbs act on it.**
- **Resource:** any "thing" worth naming: a plan, a ticket, a customer, a customer's tickets.
- **URL = the resource's address.** Nouns, not verbs: `/plans/flex-smart-ksa`, not `/getPlan?id=…`.
- **HTTP method = the verb.** The same handful of verbs work on every resource (§1's table). Learn
  them once and you can guess any REST API. This is the **uniform interface**.
- **Representation:** you never get the plan itself, only a *representation* of it, usually JSON
  (it could be XML or HTML). That's the "Representational" in the name. The client changes state by
  sending representations back (PUT/PATCH), hence "State Transfer".

**"Resources at URLs", concretely.** Think of your file system:
`/Users/wahab/Documents/cv.pdf` is the *address* of one file. Opening, renaming and deleting are
separate *actions* you apply to that address. The path never says "delete". REST works the same way:

```
https://api.telcox.example/v1/tickets/T-123?fields=status
└─────── server ────────┘└v┘└──────────┘└── options ──┘
                              the thing
```
- `/tickets` names a **collection** (all tickets); `/tickets/T-123` names **one item** in it;
  `/customers/42/tickets` names a **related collection** (customer 42's tickets).
- The address stays the same no matter what you do; **the method changes**:
  ```
  GET    /tickets/T-123     → read it
  PATCH  /tickets/T-123     → change it
  DELETE /tickets/T-123     → remove it
  ```
  One address, three actions. In RPC style the action goes *into* the URL instead:
  `/getTicket?id=T-123`, `/updateTicket`, `/deleteTicket`. That's three URLs for the same ticket.
- A resource isn't necessarily a database table. It can be anything worth naming:
  `/customers/42/usage` (computed on the fly) or `/plans/flex-smart-ksa/price-history`.
- **Why bother:** a stable address for each thing means it can be bookmarked, shared in a support
  chat ("look at /tickets/T-123"), cached (the cache key *is* the URL) and read in logs at a glance.
- **The test:** "is there a *thing* here that I could GET again tomorrow?" `/tickets/T-123`: yes.
  `/ask`: no, because nothing lives there. It's an action that computes a fresh answer every time.
  (If we stored every answer, each would get its own address, e.g. `/answers/a-789`, and *that*
  would be a resource.)

**A worked example: a RESTful API for TelcoX** (not what we're building; it's to see the pattern):

| Request | Meaning | Success code |
|---|---|---|
| `GET /plans?market=KSA` | List KSA plans (query params filter and paginate a collection) | 200 |
| `GET /plans/flex-smart-ksa` | Read one plan | 200 (404 if no such plan) |
| `POST /tickets` with `{"subject": …}` | Create a ticket; the server picks the ID | **201 Created**, plus a `Location: /tickets/T-123` header |
| `GET /tickets/T-123` | Read that ticket | 200 |
| `PATCH /tickets/T-123` with `{"status": "closed"}` | Change part of it | 200 |
| `DELETE /tickets/T-123` | Remove it | **204 No Content** |
| `GET /customers/42/tickets` | Nested: customer 42's tickets | 200 |

Notice: 2 nouns, 4 verbs, and the whole API is predictable. Compare the non-REST version:
`/getPlans`, `/fetchPlanById`, `/createNewTicket`, `/closeTicket`, `/removeTicket`… every endpoint
is a new word to learn.

**The rules that matter in practice:**
1. **Nouns in URLs, plural collections:** `/plans`, `/plans/{id}`. The method carries the verb.
2. **Use the right method and status code.** A client, a cache or a proxy can then act without
   knowing your API: GET is safe to cache, a 503 is worth retrying, a 404 isn't.
3. **Stateless.** Each request carries everything needed to serve it (including who you are, e.g.
   an `Authorization` header). The server keeps nothing *in memory* between requests. Shared state
   lives in a database. **Why:** any of 10 server copies can serve any request, so a load balancer
   can send each request anywhere, and a crashed server loses nothing.
   *Our project:* the follow-up question that failed in break-testing ("and in the UAE?") fails
   exactly because of this. Each `/ask` is independent. To support conversations, the client sends
   the history with each request, or a `conversation_id` that points to history stored in a database
   (Lesson 09). Never in a Python dict inside one server.
4. **Idempotency matters for retries.** If a network error hits a `DELETE /tickets/T-123`, the client
   can simply resend it: deleting twice has the same effect as once. Resending a `POST /tickets`
   might create two tickets. The fix is an **`Idempotency-Key`** header: the client sends a unique
   key, and the server ignores repeats of it (Stripe made this pattern famous).
5. **Version the contract:** `/v1/plans`. Breaking changes go to `/v2`, so old clients keep working.

**Strict vs practical REST.** Fielding's full definition includes **HATEOAS** (responses contain
links telling the client what it can do next), which almost nobody implements. In industry,
"RESTful" usually means "resources at URLs, proper methods and status codes, stateless, JSON".
That's the bar interviewers expect.

**So where does `POST /ask` fit?** It's **RPC style** ("do this action"), not a resource. There's no
"answer" sitting on the server to GET; we compute a new one each time. That's normal for ML and LLM
services: OpenAI's own API is `POST /v1/responses` and `POST /v1/embeddings`. We use POST rather
than GET because:
- the input is structured and can be long (a GET has no body, and URLs have length limits and end
  up in logs);
- the call isn't a cheap, cacheable read: every call costs money, and the answer can change.

*Could* we make it RESTful? Yes, by modelling conversations as resources:
`POST /conversations` → 201, then `POST /conversations/{id}/messages` with the question, and
`GET /conversations/{id}` for the history. That's worth doing once we have chat history and
storage. Until then it's ceremony with no benefit. In an interview, knowing *why* `/ask` isn't pure
REST and *when* you'd remodel it matters more than forcing it to be REST.

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
9. What is an API, and why put a model behind a web API instead of shipping it as a library?
10. What makes an API RESTful? Is `POST /ask` RESTful, and would you change it?
