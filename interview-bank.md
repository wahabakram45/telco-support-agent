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
*Project example:* "Arabic plan descriptions took about N× more tokens than the English ones, which
changed our cost estimate for the GCC market." (Fill in N after the experiment.)

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
