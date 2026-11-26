"""Score generated answers on the golden set. Lesson 06.

Run: uv run python scripts/eval_generation.py --save baseline     # generate + judge (~$0.05)
     uv run python scripts/eval_generation.py --rejudge baseline  # re-judge saved answers only
     uv run python scripts/eval_generation.py --compare baseline  # deltas against a saved run
     uv run python scripts/eval_generation.py --failures          # print every non-correct answer
"""

import argparse
import hashlib

from telco_support_agent.config import get_settings
from telco_support_agent.generation import SYSTEM_PROMPT, RAGPipeline
from telco_support_agent.generation_eval import (
    GenerationReport,
    build_report,
    generate,
    judge_all,
)
from telco_support_agent.golden import GOLDEN_PATH, load_corpus_chunks, load_golden
from telco_support_agent.judge import CORRECTNESS_PROMPT, Judge

ROWS = [
    ("correctness", "answerable: correct = 1, partial = 0.5"),
    ("correct_rate", "answerable: share fully correct"),
    ("incorrect_rate", "answerable: share wrong (incl. refusals)"),
    ("refusal_accuracy", "unanswerable: share refused"),
    ("over_refusal", "answerable: share refused"),
    ("faithfulness", "answered: no unsupported claims"),
    ("checks_ok", "answered: deterministic verify() passed"),
    ("citation_relevance", "answered: cites a relevant chunk"),
    ("cost_per_1k_usd", "generation cost per 1,000 questions"),
    ("latency_p50_s", "end-to-end latency, median"),
    ("latency_p95_s", "end-to-end latency, 95th percentile"),
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def fmt(value: float | None, width: int = 8, sign: bool = False) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>+{width}.3f}" if sign else f"{value:>{width}.3f}"


def show(report: GenerationReport, baseline: GenerationReport | None) -> None:
    s = report.summary
    print(f"{s['n']:.0f} questions   config: {report.config}\n")
    print(f"{'metric':<20}{'value':>8}   {'Δ' if baseline else ' '}       meaning")
    for key, meaning in ROWS:
        delta = " " * 8
        if baseline and s[key] is not None and baseline.summary.get(key) is not None:
            delta = fmt(s[key] - baseline.summary[key], sign=True)
        print(f"{key:<20}{fmt(s[key])} {delta}   {meaning}")
    print(f"\njudge cost this run: ${s['judge_cost_usd']:.4f}\n")
    print(f"{'category':<22}{'n':>3}{'score':>8}{'faithful':>10}")
    for cat, row in report.by_category.items():
        print(f"{cat:<22}{row['n']:>3.0f}{fmt(row['score'])}{fmt(row['faithful'], 10)}")


def show_failures(report: GenerationReport) -> None:
    bad = [r for r in report.items if r.verdict != "correct" or r.faithful is False]
    print(f"\n{len(bad)} answers not fully correct or not faithful:")
    for r in bad:
        print(f"\n  [{r.id}] ({r.category}, {r.market}) {r.question}")
        print(f"     answer:    {r.answer}")
        print(f"     reference: {r.reference}")
        print(f"     verdict:   {r.verdict} (by {r.verdict_by}): {r.reasoning}")
        if r.unsupported_claims:
            print(f"     unsupported: {r.unsupported_claims}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="LABEL", help="save the run under this label")
    parser.add_argument("--rejudge", metavar="LABEL", help="re-judge a saved run's answers")
    parser.add_argument("--compare", metavar="LABEL", help="show deltas against a saved run")
    parser.add_argument("--failures", action="store_true", help="print every non-correct answer")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    items, chunks, judge = load_golden(), load_corpus_chunks(), Judge()
    settings = get_settings()
    if args.rejudge:
        saved = GenerationReport.load(args.rejudge)
        results, config = saved.items, saved.config
    else:
        results = generate(items, RAGPipeline(), args.workers)
        config = {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "system_prompt_sha": sha(SYSTEM_PROMPT),
            "golden_sha": sha(GOLDEN_PATH.read_text()),
        }
    config = {**config, "judge_model": judge.model, "judge_prompt_sha": sha(CORRECTNESS_PROMPT)}
    results = judge_all(results, items, judge, chunks, args.workers)

    report = build_report(results, args.save or args.rejudge or "dev", config)
    baseline = GenerationReport.load(args.compare) if args.compare else None
    show(report, baseline)
    if args.failures:
        show_failures(report)
    if args.save:
        print(f"\nsaved → {report.save()}")


if __name__ == "__main__":
    main()
