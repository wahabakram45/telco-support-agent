# Lesson 05 — Retrieval evals: a golden set and the metrics that score it

> **Goal:** replace "it seems to work" with numbers. Understand how to build a golden set (and
> why its labels need as much care as code), what recall@k, precision@k, MRR and nDCG measure and
> when each matters, how sure you can be of a number, and how to read a baseline. Implement the
> core metrics, and record the baseline every later lesson must beat.
> **Time:** ~3 hours (1.5 h reading, 45 min building, 30 min writing your own questions, quiz).

So far, quality has meant 15 probe questions and gut feeling. That was fine for writing a prompt,
but it can't answer the questions that come next: *did hybrid search help? By how much? Did it
break anything?* Lessons 07–09 each change retrieval, and each will be judged by the numbers from
this lesson.

---

## Part A — Concepts

### 1. Why evals come first

- **You can't improve what you don't measure.** Without a fixed test, every change is judged on
  the 3 questions you happen to try, and those are usually the ones it fixes.
- **Regressions are silent.** A tweak that fixes Bahrain can break Flex Smart. Only a fixed set run
  every time catches that.
- **It's what interviewers probe.** "How did you know it got better?" separates people who built a
  demo from people who built a system. Your CV bullet will read "recall@5 0.88 → 0.9x after
  hybrid search", and you'll be able to defend every digit.

### 2. What to evaluate: separate retrieval from generation

A RAG answer can be wrong for two very different reasons:

```
question ──► RETRIEVAL ──► top-5 chunks ──► GENERATION ──► answer
              │                               │
      "was the right chunk found?"     "did the LLM use it correctly?"
      lesson 05 (this one)             lesson 06
```

- **Retrieval sets the ceiling.** If the right chunk isn't in the top 5, no prompt can fix the
  answer. That's why we measure it on its own first.
- **Separate scores tell you *where* to work.** "Answer correctness 70%" alone is useless. "Hit@5
  94%, but faithfulness 75%" says: stop tuning search, fix the prompt.
- **Retrieval evals are cheap and deterministic.** No LLM is involved, just embeddings, which are
  cached. The whole run costs nothing after the first time and gives the same number every time. You
  can run it on every commit.

Other distinctions you'll hear:
- **Offline vs online.** Offline = a fixed test set, before release (this lesson). Online = real
  traffic: thumbs up/down, escalations to humans, A/B tests (Phase 7).
- **Component vs end-to-end.** Retrieval metrics are component metrics. "Was the final answer
  correct?" is end-to-end. You need both.

### 3. The golden set

A **golden set** is a fixed list of questions, each with the *known* right answer source. Ours
is `data/eval/golden.yaml`, with 77 questions:

```yaml
- id: h-060
  question: Is Flex Unlimited really unlimited?
  market: KSA
  category: unlimited_fair_usage        # which trap or skill it tests
  source: hand                          # hand / synthetic / seed
  relevant:                             # what retrieval should find
    - {doc: plan-flex-unlimited-ksa}
    - {doc: fair-usage-policy, has: "1 Mbps"}
  distractors: []                       # documents that would mislead the LLM
  reference_answer: Full speed up to 150 GB per cycle, then 1 Mbps until renewal.   # for lesson 06
```

**Where questions come from**, best first:

| Source | Strengths | Weaknesses | In our set |
|---|---|---|---|
| **Real user queries** (logs, support tickets) | The true distribution: real wording, typos, real priorities | Needs a live product; privacy review; must be labelled | None yet (no users) |
| **Hand-written for known risks** | Targets the traps that matter (T1–T8), the cases you'd be embarrassed by | Slow; the author's own phrasing; biased toward what you already thought of | 49 |
| **Break-testing seeds** | Real failures you actually found | Few | 9 (from lesson 04) |
| **Synthetic (LLM-generated)** | Fast and cheap coverage of the whole corpus | Too easy, ambiguous, skewed (§8) | 19, after review |

**Slices (categories) matter more than the overall number.** "Hit@5 0.94" hides that multilingual
questions are at 0.67. Every question gets a category, and the report breaks the metrics down by
it. Our categories mirror the data card's traps T1–T8 plus paraphrase, multi-document,
multilingual and out of scope.

**Include unanswerable questions.** 7 of ours have no relevant document ("weather in Riyadh",
"how much data do I have left?"). Retrieval metrics skip them, since there's nothing to find, but
lesson 06 needs them to test that the bot refuses.

**How big?** 50–100 questions is a normal start. The limit is statistics (§7): with 70 scored
questions, a metric is only good to about ±0.05. Grow the set where the decisions are: the
categories you're actively improving.

**Treat it like code.** It's in git, reviewed, and tested (`tests/test_golden.py`). The report
records a hash of it (`golden_sha`), so two runs are only compared if they used the same questions.

**Dev vs test split.** If you tune the system on the same questions you report, you'll overfit
to them (Goodhart's law: "when a measure becomes a target, it ceases to be a good measure"). Larger
projects keep a **held-out test set** that's only run for final numbers. Ours is too small to split
yet. We note the risk and will add held-out questions as the set grows.

### 4. Labelling relevance: by content, not by chunk ID

The obvious label is "the answer is in `roaming-policy-uae#2`". It breaks as soon as you change the
chunking: `#2` becomes `#3`, or two chunks merge, and every label silently points at the wrong text.

Our labels say **"a chunk of document D whose text contains T"**:
```yaml
- {doc: roaming-policy-uae, has: "20 AED for 1 GB"}     # exactly the chunk(s) stating this fact
- {doc: plan-flex-smart-ksa}                            # no `has`: any chunk of the doc counts
```
Documents and facts are stable. The data card guarantees facts survive even when the LLM rewords
the policies, so these labels survive re-chunking and regeneration. Matching ignores case and
markdown bold.

**Graded relevance.** Not every relevant chunk is equally useful:
- **2** = answers the question;
- **1** = partial or useful context (e.g. the grace period, when the question is about balance
  validity);
- **0** = not relevant.

Binary metrics (hit, recall, precision, MRR) treat 1 and 2 alike; nDCG uses the grades.

**Labels are incomplete, and that matters.** The same fact often lives in several places: the
roaming policy *and* the roaming FAQ. If you label only one, the other counts as "irrelevant" when
retrieved, and recall is underestimated. Two defences:
- **Pooling** (from TREC, the classic IR benchmarks): pool the top results of several systems and
  judge them all, so relevant chunks that no single system found still get labelled.
- **Audit every miss.** Before believing a miss, look at what *was* retrieved. In our first run, 1
  of 5 misses was our label, not the retriever. h-021 ("Does Max Advanced roll over data?") had
  `faq-postpaid#4` at rank 2, which states the rule as "priced at 150 or more", worded differently
  from the policy's "150 SAR or 150 AED". We fixed the label. **When a miss is a labelling error, fix
  the label, not the system.**

### 5. The metrics, on one worked example

Retrieval returned 5 chunks. Their grades, in rank order: **`[0, 2, 0, 1, 0]`**, so the 2nd answers
the question and the 4th helps. The corpus holds **3** relevant chunks in total, graded `[2, 2, 1]`.

| Metric | Question it answers | Formula | Our example |
|---|---|---|---|
| **Hit@k** (hit rate) | Did *at least one* relevant chunk make the top k? | 1 if any, else 0 | top 5: **1.0**. Top 1: 0.0 |
| **Precision@k** | What share of the top k is relevant? | relevant in top k / k | 2/5 = **0.40** |
| **Recall@k** | What share of *all* relevant chunks made the top k? | relevant in top k / all relevant | 2/3 = **0.67** |
| **MRR** (mean reciprocal rank) | How high is the *first* relevant chunk? | 1 / rank of first relevant | 1/2 = **0.50** |
| **nDCG@k** | How close is the whole ranking to the ideal order, using grades? | DCG / ideal DCG | **0.43** (below) |

Each metric is computed per question, then **averaged over all questions** (so MRR is the *mean*
of the reciprocal ranks).

**nDCG step by step.** Each result earns a **gain** of `2^grade − 1` (grade 2 → 3, grade 1 → 1,
grade 0 → 0), **discounted** by its rank: divided by `log2(rank + 1)`. So rank 1 counts fully, rank
2 counts 63%, rank 5 counts 39%.
```
DCG   = 0/log2(2) + 3/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6) = 1.893 + 0.431 = 2.323
ideal = the 3 relevant chunks in the best order [2, 2, 1]:
        3/log2(2) + 3/log2(3) + 1/log2(4) = 3 + 1.893 + 0.5 = 5.393
nDCG  = 2.323 / 5.393 = 0.43
```
The ideal is "every relevant chunk at the top, best first". That's why nDCG needs the grades of
*all* relevant chunks, not just the retrieved ones.

**Which metric for what:**
- **Hit@k: "can the LLM answer at all?"** For RAG, where one good chunk is often enough, this is
  the headline metric. Pick k = what the LLM actually sees (our k = 5).
- **Recall@k: "does the LLM get *everything* it needs?"** It matters for multi-document questions.
  Our `multi_doc` slice has **hit 1.00 but recall 0.68**: the LLM always gets *something*, but often
  not the second document (e.g. the fair usage policy), so the answer will be incomplete. Hit
  alone would hide that.
- **MRR and nDCG: "is the best chunk near the top?"** Order matters because LLMs attend less to the
  middle of long contexts ("lost in the middle"), and because rerankers (lesson 08) will let us pass
  fewer chunks. nDCG also rewards putting grade-2 chunks above grade-1.
- **Precision@k: mostly noise for us.** Questions have ~1–3 relevant chunks but k = 5, so even a
  perfect retriever scores at most 1/5 – 3/5. Our 0.28 is near that ceiling. Precision matters when
  irrelevant context is costly (tokens, distraction), which is the *distractor* metric's job here.
- **MAP** (mean average precision) is another classic order-aware metric. It's common in search
  papers; nDCG is more common in industry because it handles grades.
- You'll also meet **context precision / context recall** (RAGAS and similar tools). They're the same
  ideas, but an LLM judges relevance instead of a fixed label. Lesson 06 covers LLM judges.

### 6. Distractors: irrelevant vs harmful

Precision counts every irrelevant chunk as equally bad. Some are worse: retrieving **Flex Smart
Plus** for a Flex Smart question invites the LLM to quote the wrong price. Items list such
documents as `distractors`, and **distractor@5** is the share of questions with one in the top 5.
Baseline: **0.06**, 4 questions: three from the Flex Smart / Flex Smart Plus near-duplicates
(including the T3 question, where the *Plus* page, not the 2025 version, crept in) and one from the
discontinued Flex Classic. The market and freshness filters keep the other-market and 2025
documents out completely, and this metric proves it.

### 7. How sure are you? Confidence intervals

70 questions is a small sample. If you'd written 70 *different* questions, the score would differ.
The **bootstrap** estimates by how much: resample the 70 questions with replacement 2,000 times,
compute the mean each time, and take the middle 95% of those means.

Our hit@5 is **0.943, 95% CI [0.886, 0.986]**. Practical consequences:
- A change from 0.94 to 0.95 is **one question**. Don't celebrate it, and don't panic at −0.01.
- To compare two systems, look at **which questions changed** (`--compare` and the per-item
  results), not just the average. Two systems are run on the *same* questions, a **paired**
  comparison, which is more sensitive than comparing two independent intervals.
- The intervals shrink with the square root of n: halving the uncertainty needs 4× the questions.

### 8. Synthetic questions: useful, and dangerous if unreviewed

`scripts/generate_golden.py` samples one chunk per document and asks the LLM for a question that
chunk answers, plus a short **evidence** span copied from it, which becomes the label. It cost
$0.006 for 40 questions. Then a human reviews. Here is what review found:

```
40 chunks sampled
 └─ 7 auto-rejected: the "exact" evidence span wasn't in the chunk (the model paraphrased it)
33 reviewed
 ├─ 11 dropped: AMBIGUOUS. "How much does the TelcoX KSA prepaid plan cost?" makes sense only if
 │             you've seen the passage. There are 5 prepaid plans; the label would be wrong for 4.
 ├─  3 dropped: a duplicate, a trivial T&C question, one too complex to label reliably
 ├─  3 fixed:   a missing second document; a vague name; a question needing the fair usage policy
 └─ 16 kept as-is   → 19 in the golden set
```

And the strongest lesson, from our baseline:

| Source | n | hit@5 | MRR@5 |
|---|---|---|---|
| synthetic | 19 | **1.00** | **1.00** |
| hand | 45 | 0.96 | 0.89 |
| seed (real failures) | 6 | **0.67** | 0.67 |

**Synthetic questions are too easy.** The LLM writes the question *while looking at the chunk*, so
it reuses the chunk's words ("How much does the Travel Asia roaming add-on cost every 14 days?"),
and embeddings love word overlap. A golden set made only of synthetic questions would report
near-perfect retrieval for a system that fails on "can I use my data in Bahrain?". Mitigations: ask
for paraphrases (we did, for half), write the hard cases by hand, and above all use **real user
queries** once you have them. Also watch the distribution: plan pages are 37 of 62 documents, so
random sampling over-produces price lookups.

### 9. Reading our baseline

```
hit@5        0.943  [0.886, 0.986]        hit curve  @1: 0.86  @3: 0.94  @5: 0.94  @10: 0.97
recall@5     0.880  [0.809, 0.938]
mrr@5        0.898  [0.829, 0.952]
ndcg@5       0.859  [0.788, 0.918]
distractor@5 0.057
```
- **Most questions are easy for embeddings.** Single facts and synthetic questions score ~1.0.
  Most of the remaining errors sit in a few slices.
- **The 4 misses are all known weaknesses:**
  - `seed-002` (Arabic question vs English documents: similarity ~0.2);
  - `seed-001` (Bahrain: the right chunks are at ranks 6, 7 and 10);
  - `h-011` (tourist SIM: no word overlap with "visitor prepaid line");
  - `h-071` (cheapest plan you can buy: needs comparing prices, a job for a tool in lesson 10).
- **The hit curve says the right chunk is often just below the cut** (0.94 at 5, 0.97 at 10). That's
  exactly the case where a **reranker** (lesson 08) helps: retrieve 20, reorder, keep 5.
- **multi_doc recall 0.68 and multilingual 0.39** are the slices to watch in lessons 07–09.

These go into the ROADMAP results table as **v0: recall@5 0.880, MRR 0.898**.

### 10. Our decisions

- Retrieval evaluated separately, deterministically, with the *same* market and freshness filter
  as production (`market_filter()` is shared by the pipeline and the eval).
- 77 questions: hand-written traps + break-test seeds + reviewed synthetic; 7 unanswerable.
- Content-based, graded labels; every label tested against the corpus; misses audited.
- Headline: hit@5 and recall@5 (k = what the LLM sees); MRR and nDCG for order; distractor@5 for
  harm; bootstrap 95% CIs; per-category breakdown.
- Each saved run records the embedding model, index size and golden-set hash, so comparisons are
  honest.

---

## Part B — Build

I wrote: the golden set (and its review), the label model (`golden.py`), the harness
(`retrieval_eval.py`, `scripts/eval_retrieval.py`), the synthetic generator, `hit_at_k`,
`precision_at_k`, `dcg`, the bootstrap, the data tests, and the saved baseline. **You write the
metrics at the heart of it**, in `src/telco_support_agent/retrieval_metrics.py`.

### Task 1 — recall@k and reciprocal rank (~15 min)
`recall_at_k(grades, n_relevant, k)` and `reciprocal_rank(grades)`. Any grade above 0 counts as
relevant. Ranks start at 1.

### Task 2 — nDCG@k (~20 min)
`ndcg_at_k(grades, ideal_grades, k)`, using the provided `dcg()`. The ideal ranking is
`ideal_grades` sorted best first, **cut to k** (only k slots exist). Check your understanding against
the worked example in §5 (0.43).

`uv run pytest tests/test_retrieval_metrics.py tests/test_golden.py -v` (10 tests wait for you). Then:
```bash
uv run python scripts/eval_retrieval.py --compare baseline
```
Every Δ should be **+0.000**: your implementation reproduces the saved baseline exactly.

### Task 3 — write 3 golden questions of your own (~20 min)
Add them to the end of `data/eval/golden.yaml` (ids `w-001`… to mark them as yours):
1. a **paraphrase**: a real customer's words, avoiding the document's terms;
2. a **multi-document** question that needs two documents;
3. one you **expect to fail**. Try to beat the retriever.

Find the right `doc` and `has` with `uv run python scripts/search.py "..." --market KSA`, or by
reading `data/docs/`. Then `uv run pytest tests/test_golden.py` checks your labels, and
`uv run python scripts/eval_retrieval.py --misses` shows whether they're found. Did the one you
expected to fail actually fail?

When you're done, commit and say "lesson 05 done".

---

## Part C — Interview drill

Short-format quiz on "review lesson 05". Full model answers:
[interview-bank §05](../interview-bank.md#05--retrieval-evals).

1. Why evaluate retrieval separately from generation?
2. How would you build a golden set for a RAG system with no users yet?
3. Explain recall@k, precision@k, MRR and nDCG. Which would you report for RAG, and why?
4. Why can synthetic eval questions overstate quality? How do you mitigate it?
5. Your recall@5 went from 0.86 to 0.88. Is the change real?
6. How do you label relevance so the labels survive re-chunking?
7. A question "fails" retrieval. What do you check before blaming the retriever?
8. What's the difference between a dev set and a held-out test set, and why does it matter?
