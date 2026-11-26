# Lesson 06 — Generation evals: judging answers, and judging the judge

> **Goal:** measure the *answers*, not just the retrieval: correctness, faithfulness, refusals,
> citations, cost and latency. Build an LLM-as-judge the right way, then **calibrate it against your
> own labels**, because an unvalidated judge produces confident numbers that mean nothing. Finish
> with a CI regression gate.
> **Time:** ~3.5 hours (1.5 h reading, 30 min code, 45 min labelling, 30 min calibration, quiz).

Lesson 05 showed retrieval finds a relevant chunk for 94.5% of questions. That's the ceiling, not
the result: the LLM can still pick the wrong plan from the right context, skip half the question,
or add a "fact" nobody wrote.

---

## Part A — Concepts

### 1. What can go wrong after retrieval succeeds

| Failure | Example (the kind of thing our traps provoke) |
|---|---|
| **Wrong fact from the right context** | Quotes Flex Smart Plus's price for Flex Smart (both pages retrieved) |
| **Incomplete** | Gives the price but not the validity; one of two options |
| **Unsupported claim** (hallucination) | A plausible fact in no source, e.g. a country's region or a support hour |
| **Contradiction** | "Yes, it's enough" followed by numbers that say no |
| **Over-refusal** | "I don't have that information" when the sources do answer |
| **Under-refusal** | Answers the weather question, or invents a competitor comparison |
| **Bad citations** | Cites a source that doesn't contain the claim |

Each needs a different detector. That's why "accuracy" alone is never enough.

### 2. The dimensions we measure

| Dimension | Compares the answer with | How we measure it |
|---|---|---|
| **Correctness** | The golden `reference_answer` | LLM judge: correct / partial / incorrect |
| **Faithfulness** (groundedness) | The sources the generator saw | LLM judge: list unsupported claims |
| **Refusal accuracy** | Whether the question is answerable | Rule: unanswerable questions must be refused |
| **Over-refusal** | The same | Rule: answerable questions must *not* be refused |
| **Deterministic checks** | The cited sources | Lesson 04's `verify()`: valid citations, no unsupported numbers |
| **Citation relevance** | The golden labels | Rule: does a cited chunk match the question's relevance labels? |
| **Cost, latency** | Your budget and SLA | Measured: cost per 1,000 questions, p50/p95 latency |

**Correctness and faithfulness are different questions:**
- **Correct but unfaithful:** the model guessed right from memory. It's fine today, but the next
  guess will be wrong.
- **Faithful but incorrect:** it used the sources accurately, but they were the wrong sources (a
  retrieval problem).

You'll also hear **answer relevance** (does it address what was asked?). Our correctness rubric
folds that in: an answer that ignores half the question is "partial".

### 3. Three layers: rules, LLM judges, humans

```
cheap, exact, narrow                                         expensive, slow, the ground truth
     RULES  ───────────────►  LLM JUDGE  ───────────────►  HUMANS
  status, verify(),            meaning: correct?            label a sample, to check
  citation labels              supported?                   the judge
```
**Use the cheapest layer that can actually detect the failure.**
- **Refusals are rules here.** The pipeline returns a structured `status`, so "was an unanswerable
  question refused?" needs no LLM.
- **Numbers are rules:** `verify()` already catches invented numbers.
- **Meaning needs a judge.** Is "full speed until a threshold, then slower" the same as "150 GB then
  1 Mbps"? No regex can say.
- **Humans check the judge.** You can't label every run by hand, but you can label a sample once,
  and use it to decide whether to trust the judge on every run after.

### 4. LLM-as-judge: how to build one properly

An LLM judge is just a prompt and a model, so it has all the failure modes of any LLM. Design
choices that matter:

- **One judge per question.** We use two separate calls: correctness (answer vs reference) and
  faithfulness (answer vs sources). One combined call lets one verdict contaminate the other.
- **Categorical verdicts, not 1–10 scores.** Models are poorly calibrated on fine scales: is this
  a 6 or a 7? "correct / partial / incorrect", each with a written definition, is far more
  repeatable, and humans can label it the same way.
- **Reasoning before the verdict.** The structured output has `reasoning` *first*, so the model
  thinks, then decides (a cheap form of chain-of-thought). It also makes disagreements debuggable:
  you can read *why*.
- **A rubric with concrete examples** of each verdict, written for *your* domain: "25 SAR is not 20
  SAR", "a discontinued plan offered as available is incorrect".
- **Temperature 0 and structured output** for repeatability and parseability.
- **Reference-based vs reference-free:**
  - Correctness needs a reference answer (from the golden set).
  - Faithfulness is reference-free: it needs only the sources. Reference-free checks can also run
    on live traffic, where no reference exists (Phase 7).
- **Pointwise vs pairwise.** We grade each answer on its own (pointwise). Pairwise ("which of these
  two answers is better?") is better for comparing two systems, but it suffers from **position
  bias**: judges favour whichever answer comes first. Fix it by running both orders.

**Known judge biases to watch for:**

| Bias | What happens | Mitigation |
|---|---|---|
| **Verbosity** | Longer answers rated higher | Rubric: "judge facts, not length" |
| **Self-preference** | A model rates its own outputs higher | Use a different or stronger model as the judge (`JUDGE_MODEL`) |
| **Leniency / strictness** | Consistently too generous or too harsh | **Calibration** (§6): the confusion matrix shows the direction |
| **Reference anchoring** | Treats every fact in the reference as required | Rubric wording, and references that state only what the question needs |
| **Position** (pairwise) | Prefers the first answer | Run both orders |

**Which model judges?** Ideally a stronger one than the generator. We default to the same model
(`gpt-5.4-nano`) to keep costs and config simple. `JUDGE_MODEL` in `.env` switches it, with its own
prices. The judge cost for one full run was **$0.03**.

### 5. Our correctness rubric (you'll label with it)

- **correct:** states the facts the question needs, consistent with the reference. Extra true
  detail and different wording are fine; missing a minor aside is fine.
- **partial:** right in part, but misses something the customer needs (the price but not the
  validity when both were asked, one of two options), or too vague to act on.
- **incorrect:** a key fact contradicts the reference (wrong price, plan, market or currency; a
  discontinued plan offered as available), or it refuses when the reference answers.

Refusals are graded by rule, so they never reach the judge.

### 6. Calibration: can you trust the judge?

Every judge score is a claim: "84.9% of answers are correct". That claim is only as good as the
judge's agreement with a careful human. **Calibration** measures that agreement:

1. Take a fixed set of answers (a saved run).
2. A human labels a sample **without seeing the judge's verdicts** (otherwise you anchor on them).
3. Compare: raw agreement, **Cohen's kappa**, and a **confusion matrix**.
4. If agreement is low, read the disagreements, fix the judge prompt, **re-judge the same
   answers**, and measure again.

**Why kappa and not just agreement?** Two raters who both say "correct" 90% of the time agree
about 82% of the time *by pure chance*. Kappa subtracts what chance explains:

```
kappa = (p_observed − p_chance) / (1 − p_chance)
p_chance = Σ over labels (share of rater A using it) × (share of rater B using it)
```
Worked example, 10 answers: human says correct ×6, partial ×4; judge says correct ×5, partial ×5;
they agree on 9.
```
p_observed = 9/10 = 0.9
p_chance   = 0.6 × 0.5 (both "correct") + 0.4 × 0.5 (both "partial") = 0.5
kappa      = (0.9 − 0.5) / (1 − 0.5) = 0.8
```
And the trap kappa catches: a human with 9 correct and 1 partial, against a judge that says
"correct" to everything. Raw agreement is 90%, and kappa is **0.0**: the judge is no better than
always guessing "correct".

**Reading kappa** (the usual Landis & Koch scale): < 0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate,
0.6–0.8 substantial, > 0.8 almost perfect. Even two careful *humans* rarely exceed ~0.8 on
graded verdicts, so that's the ceiling to compare against.

**The confusion matrix shows the direction.** Mass *above* the diagonal (you said correct, the
judge said partial) means the judge is too strict; below it, too lenient. A judge that's
consistently strict still *ranks* systems correctly, which is enough for comparisons. A judge that
disagrees randomly is useless.

**Our sample is enriched.** The label sheet contains every answer the judge did *not* call correct
(12), plus 12 random "correct" ones, shuffled. A purely random 24 would be ~20 easy "correct"s with
almost no signal on the hard cases. The price: agreement on this sample understates agreement
overall. Say so whenever you quote the number.

**Don't overfit the judge.** If you tune the judge prompt until it agrees with your 24 labels, those
24 stop being an honest test, just like a dev set (lesson 05). For real work, label more answers
and keep some held out.

### 7. Runs aren't repeatable

Retrieval evals give the same number every run. Generation evals don't: even at temperature 0, LLM
outputs vary slightly between calls, and the judge varies too. A rerun can move correctness by a few
points with *no* code change. Consequences:
- Compare runs on the same questions, and look at which items changed, not just the average.
- For important decisions, run 2–3 times and look at the spread.
- **Saved answers make the judge repeatable to test.** `--rejudge` grades a saved run's answers
  again, so when you change the judge prompt, only the judge changes. Calibration depends on this.

### 8. Reading our baseline

```
correctness         0.849   answerable: correct = 1, partial = 0.5   (78% correct, 14% partial, 8% incorrect)
refusal_accuracy    1.000   all 7 unanswerable questions refused
over_refusal        0.055   4 answerable questions refused
faithfulness        0.913   answered: no unsupported claims (per the judge)
checks_ok           1.000   verify(): every number traced to a cited source
citation_relevance  0.986   answered: cites a chunk the golden set marks relevant
cost_per_1k_usd     0.289   generation, gpt-5.4-nano
latency p50 / p95   2.4 s / 6.6 s   (8 requests in parallel; a single request is faster)
```

- **Over-refusals trace back to retrieval.** The 4 refused answerable questions are h-011 (tourist
  SIM), seed-001 (Bahrain), seed-002 (Arabic) and h-021 (rollover). The first three are exactly
  lesson 05's retrieval misses. In h-021 the plan page with the price sat at rank 10. In all four,
  the generator **behaved correctly given what it saw**: lesson 04's prompt said refuse rather than
  guess. Fix retrieval (lessons 07–09) and these become answers. Joining the two evals tells you
  which stage to fix.
- **`checks_ok` 1.00 vs faithfulness 0.91.** The deterministic checks catch invented *numbers*; the
  judge also catches invented *claims*. You need both layers.
- **Weakest slices:** discontinued 0.50, multilingual 0.50, multi_doc 0.58, near_duplicate 0.62.
  These are trap categories by design.
- **Faithfulness and correctness here are judge scores**, so they're only as trustworthy as the
  calibration in Task 3 says.

These become the ROADMAP results table's v0 faithfulness, correctness, p95 latency and cost.

### 9. The CI regression gate

`scripts/eval_gate.py` re-runs both evals and **exits with an error** if quality fell below the
committed baselines:

| Check | Rule | Why this tolerance |
|---|---|---|
| Retrieval hit / recall / MRR | Drop ≤ 0.02–0.03 | Deterministic: any drop is real |
| Correctness, faithfulness | Drop ≤ 0.06 | Noisy run to run (§7): tighter would fail randomly |
| Refusal accuracy | ≥ 0.85, absolute | Safety floor, whatever the baseline was |
| Over-refusal | ≤ 0.10, absolute | Usefulness floor |

`.github/workflows/ci.yml` runs on GitHub:
- **Every push and pull request:** ruff and the unit tests. Free, no key needed.
- **Pull requests only:** the eval gate, *if* the repo has an `OPENAI_API_KEY` secret. It builds the
  index, then runs the gate, for about $0.05 per run. Without the secret it skips cleanly.

Gate design rule: **a gate that fails randomly gets ignored, or deleted.** Tight tolerances for
deterministic metrics, wide tolerances plus floors for stochastic ones, and the baselines committed
to git so every change is compared with the same reference. When an improvement lands, re-save the
baseline in the same pull request.

### 10. Our decisions

- Two judges (correctness vs reference, faithfulness vs sources), categorical verdicts, reasoning
  first, temperature 0, structured output.
- Refusals and citation relevance graded by rule, numbers by `verify()`: the LLM only judges meaning.
- Answers saved with their judgments, so the judge can be changed and re-run on fixed answers.
- The judge calibrated against human labels (Cohen's kappa plus a confusion matrix) before its
  scores go on a CV.
- A CI gate with per-metric tolerances matched to each metric's noise.

---

## Part B — Build

I wrote: the judges (`judge.py`), the harness (`generation_eval.py`, `scripts/eval_generation.py`),
the calibration tooling (`judge_calibration.py`, `scripts/calibrate_judge.py`), the gate, the CI
workflow, tests, the saved baseline (`data/eval/results/generation_baseline.json`), and your label
sheet. **You write the agreement metric and provide the human labels**: the two things a judge can't
do for itself.

### Task 1 — Cohen's kappa (~15 min)
`cohen_kappa(a, b)` in `src/telco_support_agent/judge_calibration.py`. The docstring has the formula;
§6 has the worked example (0.8). `Counter` is already imported.
`uv run pytest tests/test_generation_eval.py -v` (6 tests wait for you).

### Task 2 — Label 24 answers (~45 min)
Open `data/eval/human_labels.yaml`. For each answer, replace `human: null` with `correct`,
`partial` or `incorrect`, using the §5 rubric. Add a short `note` when it's a close call.
- **Don't look at the judge's verdicts first** (don't open `generation_baseline.json`, or the fold
  below), or you'll anchor on them.
- Judge the answer against the *question* and the reference. If the reference contains more than
  the question asked for, decide whether that extra really matters to the customer.
- If you can't decide between two labels in 20 seconds, pick one and write why in `note`. Those
  close calls are exactly what the disagreements will be about.

### Task 3 — Calibrate (~15 min)
```bash
uv run python scripts/calibrate_judge.py baseline
```
Read the agreement, kappa, confusion matrix and every disagreement, with the judge's reasoning. For
each disagreement, decide who's right. Sometimes it'll be the judge, and you'll change your label.
Then answer: is the judge too strict, too lenient, or random? Would you trust its correctness number
on your CV?

### Task 4 — Improve the judge, together (~30 min)
Based on the disagreements, change `CORRECTNESS_PROMPT` in `judge.py`, then re-judge the **same**
answers and re-calibrate:
```bash
uv run python scripts/eval_generation.py --rejudge baseline --save judge-v2
uv run python scripts/calibrate_judge.py judge-v2
```
Tell me what you want to change, and we'll iterate the way we did with the system prompt in lesson
04.

<details>
<summary>Read only after Task 2: what I noticed in the judge's reasoning</summary>

Reading the baseline's failures before making your label sheet, I saw patterns worth testing:
- **Reference anchoring.** For "How much data does Flex Smart give?", the answer "12 GB" was
  marked partial because the *reference* also mentions the price. The question didn't ask for it.
- **Reasoning and verdict disagreeing.** One judgment's reasoning ends "this is at most a minor
  omission", and then the verdict is "partial". The rubric says minor omissions are fine.
- **Faithfulness false positives.** The faithfulness judge flagged "the sources don't mention the
  brand TelcoX", and "the 1 Mbps rule applies specifically to Max Elite" (the fair usage policy
  covers every unlimited plan, so it's supported). Both are pedantic misses.
- **Real catches the rules can't make.** One answer called Turkey "a GCC country", which no source
  says. Another said "Yes, Flex Start is enough" right before showing it isn't. The judge caught
  both; `verify()` couldn't, since there's no wrong number in either.

Your labels will show whether these are one-offs or patterns.
</details>

### Optional — turn on the CI gate
Pushing this lesson starts the `lint-and-test` job (the GitHub repo's **Actions** tab). The eval
gate runs on pull requests only if you add an `OPENAI_API_KEY` secret (repo Settings → Secrets and
variables → Actions). That's your decision: it puts your key in GitHub, and each PR run costs about
$0.05. Ideally use a separate key with a low spend limit.

When you're done, say "lesson 06 done".

---

## Part C — Interview drill

Short-format quiz on "review lesson 06". Full model answers:
[interview-bank §06](../interview-bank.md#06--generation-evals).

1. How do you evaluate a RAG system's answers? Which dimensions, and how is each measured?
2. Faithfulness vs correctness: what's the difference, and why measure both?
3. How do you design a reliable LLM judge?
4. What biases do LLM judges have, and how do you mitigate them?
5. How do you know your LLM judge is trustworthy?
6. Why Cohen's kappa instead of raw agreement?
7. Your eval scores change between runs with no code change. Why, and what do you do about it?
8. How would you put evals in CI without making the pipeline flaky?
