# Lesson 00b — LLM API Fundamentals

> **Goal:** understand what actually happens in one LLM API call (tokens, context, sampling, cost,
> structured output) and build the small wrapper that every later phase will use.
> **Time:** ~5 hours (1.5 h reading, 3 h building, 30 min drill).

Every RAG system, agent and eval in this project comes down to *"send a list of messages, get text
back, pay per token."* If you understand that one call deeply, you can reason about cost, latency
and failures. Interviewers test this first.

---

## Part A — Setup (~15 min)

Do [Lesson 00a](00a-python-toolkit.md) first. It installs uv, git and pre-commit, and teaches the
Pydantic you'll need here. Then:

```bash
cp .env.example .env          # .env is gitignored and never committed
```

**OpenAI account:**
1. Go to platform.openai.com → Billing. Add $10 credit and **set a monthly usage limit (e.g. $15)**.
   That's a real production habit, and it's worth mentioning in interviews.
2. Create a project API key and put it in `.env` as `OPENAI_API_KEY`.
3. Open the official pricing page (openai.com/api/pricing). Pick the **cheapest current "mini" or "nano"
   chat model** and write its name and its input/output/cached-input prices into `.env`.
   Third-party price lists disagree, so always use the official page.

Check: `uv run pytest` should show the unit tests failing with `NotImplementedError`. That's correct,
because you haven't written the code yet.

---

## Part B — Concepts

### 1. What an API call actually is

An LLM is a function: **sequence of tokens in → probability distribution over the next token out.**
"Chat" is just a formatting convention on top of that. Your messages are serialized into one long
token sequence with role markers:

```
[system]    You are a support assistant for TelcoX. Only answer from the provided documents.
[user]      Which bundle has the most data under 50 SAR?
[assistant] ...the model generates from here, one token at a time...
```

- **system / developer message**: standing instructions (persona, rules, output format). It has
  higher priority than user text, but it's not a security boundary (see prompt injection, Lesson 14).
- **user**: the human's input. In RAG, the retrieved documents also go here (or into the system message).
- **assistant**: the model's previous replies. **The API is stateless.** The model remembers nothing
  between calls. "Conversation memory" means *you* resend the whole history every time, and you pay
  for it every time. This is why long chats get expensive and why Lesson 09 covers query rewriting.

### 2. Tokens

Models don't see characters or words. They see **tokens**: sub-word pieces produced by a tokenizer
(BPE, byte-pair encoding). Rules of thumb for English: 1 token ≈ 4 characters ≈ 0.75 words.

```
"roaming"        → ["ro", "aming"]        (2 tokens)
"SAR 49.99/month" → ["S","AR"," ","49",".","99","/","month"]   numbers and symbols are expensive
```

Why it matters:
- **You pay per token**, and input and output have different prices (output usually costs 4–8× more).
- **The context window limit is counted in tokens** (prompt + documents + history + output).
- **Latency is mostly output tokens.** Input is processed in parallel ("prefill"). Output is generated
  one token at a time ("decode"). A 2,000-token answer is slow, while a 2,000-token prompt is cheap and fast.
- **Token efficiency differs by language and tokenizer.** Older tokenizers spent 2–4× more tokens on Arabic
  or Urdu than on English. The current `o200k_base` closed most of that gap: in our run, an Arabic usage
  sentence was 14 tokens vs 15 in English. Code and long numbers are still token-hungry. Measure your
  own traffic instead of trusting a rule of thumb.
- Tokenization causes classic failures: miscounting letters in a word, arithmetic on long numbers.
  That's one reason **we never let the LLM compute prices**. Tools do the math (Phase 5).

You'll use `tiktoken` (OpenAI's tokenizer library) to count tokens *before* sending a request.

### 3. Context window

The maximum number of tokens (input + output) per call, often 128k–1M today. A large window does
**not** mean you should fill it:
- **Cost** scales linearly with input tokens, on every call.
- **"Lost in the middle"**: models attend best to the start and end of a long context and miss facts
  buried in the middle. Retrieval quality beats stuffing.
- **Latency** grows with prompt size.

This is the core argument for RAG over "just paste all the docs": cheaper, faster, more accurate,
and the source can be cited. It's a very common interview question (see the drill).

### 4. Sampling: temperature, top_p, max tokens

At each step the model outputs probabilities for every possible next token. Sampling picks one.
- **temperature** scales how flat or peaked that distribution is. Near 0 the model almost always
  picks the top token (focused, repeatable). Higher values (0.8–1.2) give more variety.
- **top_p** (nucleus sampling) samples only from the smallest set of tokens whose probabilities add up
  to p. Tune temperature *or* top_p, not both.
- **max_output_tokens** is a hard cap on the reply length, which also caps cost.

**Our choice:** low temperature (0–0.2) for answering and for the eval judge, because we want factual,
repeatable output. Note that **temperature 0 is still not fully deterministic** (GPU floating-point
order, batching, model updates). That's why evals run over a whole set and track averages rather than
trusting one run. Some reasoning models fix the temperature and ignore it. Check the model's docs.

**Why temperature 0 still varies, step by step** (we demonstrated this live: the same story prompt
sent 5× at temperature 0 gave 5 different stories, all starting "The SIM card lived in a…" before splitting):
1. Temperature 0 = always pick the top-scoring token. That's deterministic *only if the scores are bit-identical*.
2. Floating-point addition depends on order: `(0.1 + 0.2) + 0.3 == 0.6000000000000001`, but
   `0.1 + (0.2 + 0.3) == 0.6`. Summing 1M numbers in 3 orders gave 3 different totals (~1e-12 apart).
3. GPUs add in parallel chunks with no fixed order, and **batching** (your request grouped with other
   users' requests) changes how the work is split. So the scores depend slightly on server load.
4. When two tokens nearly tie, a 6th-decimal difference flips the winner. Every later token is
   predicted from different text, so the rest of the answer diverges (butterfly effect).
5. Other sources: silent model updates behind the same name, mixture-of-experts routing that depends
   on the batch, provider-side sampling.

Factual tasks barely vary (when the answer is "2 GB", the top token wins by a mile); creative text has
near-ties at almost every word.

### 4b. Reasoning models: what changes (applies to our model, gpt-5.4-nano)

Most current models are **reasoning models**. Before answering, they generate hidden "thinking"
tokens. Three consequences:
- **Hidden tokens are billed as output.** `usage.output_tokens` includes them, and
  `usage.output_tokens_details.reasoning_tokens` tells you how many there were. A 50-word answer can
  cost hundreds of output tokens. Log them, because they're part of your cost per query.
- **The control is reasoning effort, not temperature.** The API takes a `reasoning={"effort": ...}`
  parameter (low effort = fewer hidden tokens, cheaper and faster). Many reasoning models **reject
  `temperature`**. If `chat()` fails with "unsupported parameter", that's why: drop `temperature` and
  set a low effort instead. Check the model page for which parameters it supports. (In our run,
  gpt-5.4-nano *accepted* `temperature=0` and used few or no hidden tokens at default settings.
  Behaviour differs per model, so verify it with a real call.)
- **`max_output_tokens` caps reasoning + answer together.** Set it too low and the model can use the
  whole budget thinking and return an *empty* answer. That's a classic production bug.

Interview angle: "For simple extraction and grounded RAG answers I used low reasoning effort. Higher
effort cost more and added latency without improving my eval scores." (You'll measure this in Phase 3.)

### 5. Cost math (you'll implement this)

```
cost = (uncached_input_tokens × input_price
      + cached_input_tokens   × cached_price
      + output_tokens         × output_price) / 1,000,000
```

Prices are per **million** tokens. **Prompt caching:** if the beginning of your prompt (the prefix) is
identical to a recent request, the provider reuses its computation and charges a large discount on
those tokens. OpenAI does this automatically above a minimum prompt length. Design rule:
**put static content first** (system prompt, tool definitions, examples) **and variable content last**
(retrieved docs, user question). We'll measure the effect in Phase 6.

Back-of-envelope estimate you should be able to do live in an interview:
> 1,000 queries × (3,000 input tokens + 300 output tokens). With a model priced at $0.40 input and
> $1.60 output per million tokens: 3M × 0.40 + 0.3M × 1.60 = $1.20 + $0.48 = **$1.68 per 1k queries**.

### 6. Structured outputs ➕ (a QuantCo JD gap)

Free-text output is fine for humans and awful for code. If you parse a JSON blob from free text,
it'll eventually break (missing brace, extra commentary, a field invented as `"data": "about 2GB"`).
Three levels of rigor:

| Approach | How | Guarantee |
|---|---|---|
| Prompt only | "Reply in JSON with keys …" | None; breaks occasionally |
| JSON mode | Model is constrained to emit *valid JSON* | Valid syntax, but any shape |
| **Structured outputs (schema)** | You pass a JSON Schema / Pydantic model; decoding is **constrained** so tokens that would break the schema can't be sampled | Valid **and** matches your schema |

**Our choice:** structured outputs with Pydantic models. You define the class once, the SDK turns it
into a JSON Schema, and you get a validated Python object back. Remember that a schema guarantees the
**shape**, not the **truth**. The model can still put `data_gb=20` when the user said 2. Correctness is
what evals are for.

This is also how **tool calling** works under the hood (Phase 5): a tool call is just structured
output where the schema describes a function's arguments.

### 7. Why wrap the API at all?

Calling the SDK directly works, but every call in this project needs the same extras: token and cost
logging, retries on rate limits (429) and server errors (5xx), timeouts, and the model name from
config. A thin wrapper gives you:
- **one place** to measure cost per query (a CV number),
- **provider-swappable** code (OpenAI today, maybe Claude or a local model later),
- a seam for **mocking in tests** and for adding **tracing** in Phase 6.

We are deliberately **not** using LangChain yet. Building it yourself first means you'll know exactly
what the framework does when we use LangGraph in Phase 7. "I built it raw, then moved to a framework,
and here's what I gained and lost" is a strong interview answer.

---

## Part C — Build (~3 h)

Files are in `src/telco_support_agent/`. Claude wrote the glue. You fill in the `# TODO(you)` parts.

1. **`pricing.py → estimate_cost()`**: the formula from §5. Pure function. Make `tests/test_pricing.py` pass.
2. **`llm.py → count_tokens()`**: use `tiktoken`. Make `tests/test_llm.py::test_count_tokens*` pass.
3. **`llm.py → LLMClient.chat()`**: call the OpenAI **Responses API** (`client.responses.create`), read
   `response.usage`, build a `CallRecord`, and log it.
   Docs: platform.openai.com/docs/api-reference/responses
4. **`llm.py → LLMClient.extract()`**: structured output with `client.responses.parse(..., text_format=Model)`.
   Docs: platform.openai.com/docs/guides/structured-outputs
5. **`schemas.py → UsageProfile`**: already defined. Read it. It's the input to the
   `recommend_bundle` tool you'll build in Phase 5.
6. Run `uv run python scripts/hello_llm.py` and look at the token and cost log.

**Experiments to run and note down** (they come up in interviews):
- Send the same question 5 times at temperature 0 and 5 times at 1.0. How different are the answers?
- Ask the same thing in English and in Arabic or Urdu. Compare the token counts.
- Give `extract()` a vague message ("I use a lot of data, barely call"). What does the model fill in?
  Should `data_gb` be optional? (Hint: yes. Make the model say "unknown" rather than invent a number.
  That's your first guardrail.)

When you're done, tell me "review lesson 00b". I'll review the code, then quiz you.

---

## Part D — Interview drill

These are the questions you should be ready for. When you say "review lesson 00b", I quiz you in **short format** (multiple choice, one-liners, quick estimates), then show the full model answers.
Full answers are in [`interview-bank.md` §00b](../interview-bank.md#00b--llm-api-fundamentals).

1. What is a token, and why do LLM providers charge by token rather than by request?
2. Why is output more expensive and slower than input?
3. We have a 1M-token context window. Why not just put all the documentation in the prompt instead of building RAG?
4. Is temperature 0 deterministic? How does that affect how you evaluate an LLM system?
5. How would you make an LLM return reliably parseable data? What still can go wrong?
6. Estimate the monthly cost of a support bot: 50k queries per day, 2.5k input / 250 output tokens per query, $0.40 / $1.60 per million tokens.
7. What is prompt caching and how do you structure prompts to benefit from it?
8. The API is stateless. How does a chatbot "remember" the conversation, and what problems does that cause?
