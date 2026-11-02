# Lesson 01 — Data for RAG

> **Goal:** design and build the dataset every later phase runs on: a structured plan catalog plus
> ~65 markdown documents with metadata and **deliberate traps** for your evals to catch.
> **Time:** ~5–6 hours (1.5 h reading, 3.5 h building, 30 min quiz).

A RAG system can never be better than its data. Retrieval can only find what's in the corpus, and the
LLM can only be as correct as what retrieval hands it. Most real RAG projects spend more time on data
than on models, and interviewers know it. "Tell me about your data pipeline" is often the first question.

---

## Part A — Concepts

### 1. Structured vs unstructured data, and where each one goes

| | Structured | Unstructured |
|---|---|---|
| Looks like | Rows and fields: `price=49.0, data_gb=10` | Prose: policies, FAQs, T&Cs |
| Good at | Exact lookups, filters, math, comparisons | Explanations, rules, "how do I…", "why…" |
| Accessed by | Code: SQL, a function, **a tool** (Phase 5) | **Retrieval** (RAG) |
| Failure if misused | — | LLM misreads a number in a table or does arithmetic wrong |

**The routing rule:** questions that need **exact facts, numbers, filters or calculations** should be
answered from structured data through code. Questions that need **explanation or policy** go to RAG.
"Which plan under 60 SAR has the most data?" is a sort-and-filter: an LLM reading 40 retrieved chunks
will get it wrong some of the time, and 3 lines of Python never will.

**Our design:**
- `data/catalog/plans.json` is the **single source of truth** for plans: prices, allowances, dates. The
  `search_plans` and `recommend_bundle` tools in Phase 5 read it directly.
- One markdown page per plan is **rendered from that same JSON**, so RAG can answer "what's included
  in Flex 50?". Because the pages are generated, they can never drift out of sync with the catalog.
- Policies, FAQs and T&Cs are markdown documents with no structured equivalent. RAG handles those.

This is also where the README's guardrail "**never invent prices**" really starts: prices live in
structured data, and the LLM quotes them from a tool result, not from memory.

### 2. Metadata: the data about each document

Every document carries a small header of fields besides its text. Ours:

| Field | Example | Used for |
|---|---|---|
| `doc_id` | `roaming-policy-ksa` | Citations, eval ground truth ("the answer is in doc X") |
| `doc_type` | `plan` / `policy` / `faq` | Filtering, routing, debugging |
| `market` | `KSA` / `UAE` / `ALL` | **Filtering**: a KSA customer must never get UAE prices |
| `plan_ids` | `[flex-50-ksa]` | Linking plan pages to the catalog |
| `effective_from` / `effective_to` | `2026-07-01` / `null` | **Freshness**: hide or down-rank outdated versions |
| `superseded_by` | `roaming-policy-ksa` | Pointing from an old version to its replacement |

Why it matters:
- **Filtering.** Restrict search to `market in (KSA, ALL)` before or after retrieval. That's far more
  reliable than hoping the embedding notices "KSA" (Lesson 09).
- **Citations.** The answer can say *which* document it came from. That builds user trust, and it's
  how you check faithfulness in evals.
- **Freshness and conflicts** (§6).
- **Access control** in real systems: an HR bot must not retrieve documents the user can't see.
  Filtering on permission metadata is the standard approach.
- **Debugging.** When an answer is wrong, metadata tells you *what* was retrieved and why.

Rule of thumb: **capture metadata at ingestion, while you still know it.** Recovering it afterwards
from chunks of text is painful.

### 3. Document format

Our documents are **markdown with YAML frontmatter** (the metadata block between `---` lines at the top):

```markdown
---
doc_id: roaming-policy-ksa
doc_type: policy
market: KSA
effective_from: 2026-07-01
---

# Roaming Policy (KSA)
...
```

It's human-readable, diffable in git, keeps metadata next to the text, and markdown headings give
chunking natural boundaries (Lesson 02).

**In real jobs, parsing is often the hardest part.** Company knowledge lives in PDFs, scanned images,
HTML pages, Word files and slide decks. Tables lose their structure, multi-column PDFs get read in the
wrong order, headers and footers repeat on every page, and scans need OCR. Tools like `unstructured`,
`docling` or PyMuPDF help, but expect to inspect the output by hand. Interviewers love asking about
this. Your honest answer is that you used clean synthetic markdown on purpose, to focus on retrieval
and evals, and that you know where real ingestion gets hard.

### 4. Real vs synthetic data

| | Real public pages | Synthetic (our choice) |
|---|---|---|
| Realism | High | Only as good as your design |
| Ground truth | Unknown: you'd have to verify every fact | **Known exactly**, because you wrote the facts |
| Traps for evals | Whatever happens to exist | **Designed on purpose** |
| Legal | Terms of service, copyright, scraping rules | None |
| Stability | Pages change and break your evals | Fixed and versioned in git |

We use a fictional operator, **TelcoX**, in two markets, **KSA (SAR)** and **UAE (AED)**. The two
markets are there to create real-feeling conflicts. And per the project rule: **no real company data,
ever.**

**Using an LLM to write synthetic data: the risks and how we handle them**
- **Invented numbers.** Ask an LLM for a roaming policy and it'll make up rates. We give it a fixed
  list of **facts** per document, and you'll write a checker that flags any number in the output
  that isn't in the facts. That checker is a tiny preview of faithfulness evals (Phase 3).
- **Sameness.** LLM text tends to sound alike. Different document types and styles help.
- **Eval leakage.** If you later generate eval questions from the same text with the same model, the
  questions may be suspiciously easy to retrieve. In Phase 3 we write questions from the *facts*, and
  some by hand.

### 5. Designing traps

A clean dataset produces flattering numbers and teaches you nothing. Each trap below targets a
specific, realistic failure, and in Phase 3 each one becomes an **eval category** with its own score.

| Trap | Example | Failure it catches |
|---|---|---|
| **T1 Near-duplicate names** | "Flex 50" vs "Flex 50 Plus" | Retrieval or answer mixes up similar plans |
| **T2 Regional variants** | "Max 150" exists in KSA and in UAE at different prices | Ignoring the market → wrong currency or price |
| **T3 Price history** | Old version of a plan (ended 2026-06-30) and a new one at a different price | Quoting an outdated price |
| **T4 Unlimited + fair usage** | "Unlimited" data, full speed only up to a threshold | Overselling "unlimited" |
| **T5 Discontinued plan** | No longer sold, no replacement | Recommending something you can't buy |
| **T6 Superseded policy** | Roaming policy 2025 vs 2026 with different rates | Stale document wins over the current one |
| **T7 Market-specific rules** | VAT **included** in KSA prices, **added on top** in UAE | Applying one market's rule to the other |
| **T8 Buried fact** | One clause in a ~2,000-word T&C | Lost-in-the-middle, chunking that splits a fact |

You'll build T1–T5 in the catalog. I've built T6–T8 into the document topics.

### 6. Freshness and conflicting documents

Real corpora always contain outdated versions. Strategies, roughly from simplest to most robust:
1. **Delete** old versions at ingestion. Simple, but you lose history ("what was my old price?").
2. **Filter** to current documents by default (`effective_to is null`). Our default.
3. **Down-rank** older versions so they're still retrievable but lose ties.
4. **Show both and tell the LLM** which one is current, using the metadata in the prompt.

We keep old versions in the corpus on purpose, so we can **measure** how often stale documents leak
into answers before and after adding filtering (Lesson 09). That gives you a before/after number.

### 7. PII and licensing

Support data in real companies is full of personal data (names, numbers, IDs, call transcripts). Before
anything is embedded: minimize what you ingest, mask or remove PII, enforce access control through
metadata, and remember that **embeddings are not anonymization**: text can be partly reconstructed
from vectors. And check licenses before using any external content.

---

## Part B — Build (~3.5 h)

New files: `catalog.py`, `documents.py`, `checks.py` in `src/telco_support_agent/`, and two scripts.
Claude wrote rendering, loading, file handling and all tests. You write the parts marked `# TODO(you)`.

### Task 1 — The `Plan` model (`catalog.py`, ~45 min)
Write the Pydantic model. **Field names are fixed** (rendering code depends on them). Types,
constraints and validators are yours. `tests/test_catalog.py` checks the rules.

| Field | Meaning | Rule |
|---|---|---|
| `plan_id` | `flex-50-ksa` | lowercase letters, digits, hyphens only (`pattern=`) |
| `name` | `Flex 50` | |
| `market` | `KSA` / `UAE` | use the `Market` type |
| `currency` | `SAR` / `AED` | **must match the market** |
| `plan_type` | `prepaid` / `postpaid` / `data_only` / `roaming_addon` | use `PlanType` |
| `price` | `49.0` | > 0 |
| `validity_days` | `28` | > 0 |
| `data_gb` | `10` | ≥ 0; **must be `None` exactly when `unlimited_data` is true** |
| `unlimited_data` | `False` | default `False`; if true, `fair_usage_gb` is required |
| `fair_usage_gb` | `100` | optional, ≥ 0 |
| `local_minutes`, `sms` | `300`, `100` | ≥ 0 |
| `roaming_countries` | `["UAE", "Bahrain"]` | default empty; **required for `roaming_addon`** |
| `roaming_data_gb` | `5` | optional, ≥ 0 |
| `features` | `["5G", "Free Anghami"]` | default empty |
| `effective_from` | `2026-01-01` | a `date` |
| `effective_to` | `2026-06-30` / `None` | optional; if set, must be after `effective_from` |
| `status` | `active` / `discontinued` | default `active` |
| `replaced_by` | `flex-50-ksa-v2` | optional; **only allowed if `effective_to` is set** |

Rules involving **several fields** need `@model_validator(mode="after")` instead of `@field_validator`.
Look it up in the Pydantic docs, because that's the main new concept in this task.

Design point: notice we use an explicit `unlimited_data` flag rather than "`data_gb=None` means
unlimited". In `UsageProfile`, `None` means *unknown*. Using `None` to mean two different things in two
models is a classic source of bugs. **Make states explicit.**

### Task 2 — Build the catalog (`scripts/generate_plans.py → build_plans()`, ~1.5 h)
Return a list of `Plan` objects:
- **≥ 30 current plans** covering both markets and all four plan types in each market.
- Traps **T1–T5** from §5. `tests/test_catalog_data.py` checks each one.
- Realistic numbers: prepaid 28-day plans roughly 30–300 SAR/AED, postpaid 30-day roughly 100–500, and
  so on. Invent your own naming scheme ("Flex", "Max", "DataGo", "Travel"…).

You can write the plans as a literal list, or generate tiers with a loop and add traps by hand.
Either is fine. Then run `uv run python scripts/generate_plans.py`. It writes `plans.json` and one
markdown page per plan version. Open a few pages and check they read well.

### Task 3 — The number checker (`checks.py → find_unsupported_numbers()`, ~30 min)
Return the numbers in a generated text that don't appear in its source facts. The spec is in the
docstring, and `tests/test_checks.py` shows the edge cases: `1,000` == `1000`, `50.0` == `50`, `15%`
counts as `15`, and single digits are ignored (list numbering). Hint: `re.findall`.

### Task 4 — The generation prompt (`scripts/generate_docs.py → build_prompt()`, ~45 min)
I've defined 25 document topics with their facts. Write the prompt that turns one topic's facts into
a realistic document. Things to think about:
- **Role and audience:** who's writing, for whom?
- **Format per `doc_type`:** policies get headings and short sections; FAQs are question/answer pairs.
- **Constraint:** use only the given facts; never introduce a number that isn't in them.
- **Length:** each topic has a `target_words`.
- **Output:** markdown starting with a `# Title` heading. No frontmatter, because the script adds it.

Run `uv run python scripts/generate_docs.py`. It skips documents that already exist (pass `--force`
to regenerate), retries when your checker finds unsupported numbers, and prints warnings. Cost is about
$0.05 for all 25. If the checker keeps flagging numbers like `4.1` or `10` in the long T&C, look
at where they come from: that's a prompt problem, not a checker problem. **Read at least 5 generated documents.** Did the model follow your instructions?
Tweak the prompt and regenerate the bad ones. This is prompt iteration, the core loop of the job.

### Task 5 — Check everything loads
`uv run pytest` should pass completely. `tests/test_documents.py` loads the whole corpus and checks
for duplicate IDs and broken `superseded_by` links.

When you're done, commit and say "lesson 01 done". I'll review, clean up, write the data card
(`data/DATA_CARD.md`, documenting the dataset and its traps), and quiz you.

---

## Part C — Interview drill

When you say "review lesson 01", I quiz you in **short format**. Full model answers are in
[interview-bank §01](../interview-bank.md#01--data-for-rag).

1. Which questions should a RAG system answer from documents, and which from structured data?
2. Why does metadata matter in RAG? Give three uses.
3. How do you handle outdated or conflicting documents in a RAG corpus?
4. Pros and cons of synthetic data for building and evaluating RAG?
5. What makes ingesting real enterprise documents hard?
6. How do you stop a support bot from quoting wrong prices?
7. How do you handle PII in a RAG corpus?
8. Documents change daily. How do you keep the index fresh?
