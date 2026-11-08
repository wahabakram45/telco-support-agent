"""Score retrieval on the golden set. Lesson 05.

Run: uv run python scripts/eval_retrieval.py                     # print the report
     uv run python scripts/eval_retrieval.py --misses            # what was retrieved for each miss
     uv run python scripts/eval_retrieval.py --save baseline     # save to data/eval/results/
     uv run python scripts/eval_retrieval.py --compare baseline  # deltas against a saved run
"""

import argparse
import json

from telco_support_agent.golden import load_corpus_chunks, load_golden
from telco_support_agent.retrieval import Retriever
from telco_support_agent.retrieval_eval import METRICS, RESULTS_DIR, RetrievalReport, evaluate


def show(report: RetrievalReport, baseline: RetrievalReport | None) -> None:
    k = report.k
    print(
        f"Golden set: {report.n_items} questions, {report.n_scored} scored "
        f"({len(report.skipped)} unanswerable skipped)   config: {report.config}\n"
    )
    print(f"{'metric':<14}{'value':>7}   95% CI          " + ("Δ vs baseline" if baseline else ""))
    for m in METRICS:
        lo, hi = report.ci95[m]
        line = f"{f'{m}@{k}':<14}{report.overall[m]:>7.3f}   [{lo:.3f}, {hi:.3f}]"
        if baseline:
            line += f"   {report.overall[m] - baseline.overall[m]:+.3f}"
        print(line)
    curve = "  ".join(f"@{n}: {v:.2f}" for n, v in report.hit_curve.items())
    print(f"\nhit curve      {curve}\n")

    cols = ["hit", "recall", "mrr", "ndcg", "distractor"]
    print(f"{'category':<22}{'n':>3}" + "".join(f"{c:>11}" for c in cols))
    for cat, row in report.by_category.items():
        print(f"{cat:<22}{row['n']:>3.0f}" + "".join(f"{row[c]:>11.2f}" for c in cols))


def show_misses(report: RetrievalReport) -> None:
    misses = [r for r in report.items if r.hit == 0]
    print(f"\n{len(misses)} misses (nothing relevant in the top {report.k}):")
    for r in misses:
        print(f"\n  [{r.id}] ({r.category}, {r.market}) {r.question}")
        print(f"      relevant chunks in corpus: {r.n_relevant}")
        for i, x in enumerate(r.retrieved, start=1):
            mark = "✓" if x.grade else ("⚠ distractor" if x.distractor else "")
            print(f"      {i:>2}. {x.score:.3f}  {x.chunk_id:<40} {mark}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--save", metavar="LABEL", help="save the run under this label")
    parser.add_argument("--compare", metavar="LABEL", help="show deltas against a saved run")
    parser.add_argument("--misses", action="store_true", help="print every miss in detail")
    args = parser.parse_args()

    report = evaluate(
        load_golden(), Retriever(), load_corpus_chunks(), k=args.k, label=args.save or "dev"
    )
    baseline = None
    if args.compare:
        path = RESULTS_DIR / f"retrieval_{args.compare}.json"
        baseline = RetrievalReport(**json.loads(path.read_text()))
    show(report, baseline)
    if args.misses:
        show_misses(report)
    if args.save:
        print(f"\nsaved → {report.save()}")


if __name__ == "__main__":
    main()
