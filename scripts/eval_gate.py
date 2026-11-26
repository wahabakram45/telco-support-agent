"""Regression gate: re-run the evals and fail (exit 1) if quality dropped below the baselines.
Lesson 06. Runs in CI on pull requests when an OPENAI_API_KEY secret is configured.

Run: uv run python scripts/eval_gate.py                    # retrieval + generation (~$0.05)
     uv run python scripts/eval_gate.py --retrieval-only   # free after the first run (cached)

Retrieval is deterministic, so any drop is real: tight tolerances. Generation varies run to run
(sampling, judge noise), so its tolerances are wider, plus absolute floors for safety metrics.
"""

import argparse
import sys

from telco_support_agent.generation import RAGPipeline
from telco_support_agent.generation_eval import (
    GenerationReport,
    build_report,
    generate,
    judge_all,
)
from telco_support_agent.golden import load_corpus_chunks, load_golden
from telco_support_agent.judge import Judge
from telco_support_agent.retrieval import Retriever
from telco_support_agent.retrieval_eval import RESULTS_DIR, RetrievalReport, evaluate

# metric: largest allowed drop vs the baseline
RETRIEVAL_MAX_DROP = {"hit": 0.02, "recall": 0.02, "mrr": 0.03}
GENERATION_MAX_DROP = {"correctness": 0.06, "faithfulness": 0.06}
# metric: (direction, bound). Absolute limits, whatever the baseline was.
GENERATION_FLOORS = {"refusal_accuracy": (">=", 0.85), "over_refusal": ("<=", 0.10)}


def check_drops(name: str, current: dict, baseline: dict, max_drop: dict) -> list[str]:
    failures = []
    for metric, allowed in max_drop.items():
        drop = baseline[metric] - current[metric]
        status = "FAIL" if drop > allowed else "ok"
        print(
            f"  {status:<4} {name}.{metric}: {current[metric]:.3f} "
            f"(baseline {baseline[metric]:.3f}, allowed drop {allowed})"
        )
        if drop > allowed:
            failures.append(f"{name}.{metric} dropped {drop:.3f} > {allowed}")
    return failures


def check_floors(current: dict) -> list[str]:
    failures = []
    for metric, (direction, bound) in GENERATION_FLOORS.items():
        value = current[metric]
        ok = value >= bound if direction == ">=" else value <= bound
        status = "ok" if ok else "FAIL"
        print(f"  {status:<4} generation.{metric}: {value:.3f} (must be {direction} {bound})")
        if not ok:
            failures.append(f"generation.{metric} = {value:.3f}, must be {direction} {bound}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    items, chunks = load_golden(), load_corpus_chunks()
    failures: list[str] = []

    print("retrieval")
    baseline = RetrievalReport.model_validate_json(
        (RESULTS_DIR / f"retrieval_{args.baseline}.json").read_text()
    )
    current = evaluate(items, Retriever(), chunks)
    failures += check_drops("retrieval", current.overall, baseline.overall, RETRIEVAL_MAX_DROP)

    if not args.retrieval_only:
        print("generation")
        gen_baseline = GenerationReport.load(args.baseline)
        results = judge_all(generate(items, RAGPipeline()), items, Judge(), chunks)
        gen = build_report(results, "gate", {}).summary
        failures += check_drops("generation", gen, gen_baseline.summary, GENERATION_MAX_DROP)
        failures += check_floors(gen)

    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\ngate passed")


if __name__ == "__main__":
    main()
