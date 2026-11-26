"""Run the golden set through the full RAG pipeline and score the answers. See lesson 06.

Two stages, saved together so the second can be re-run alone:
1. generate: ask the pipeline every golden question (costs generation calls);
2. judge: deterministic checks + the LLM judges on those fixed answers. Re-judging saved answers
   is how you iterate on the judge without the answers changing underneath you.

Run: uv run python scripts/eval_generation.py
"""

import json
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from telco_support_agent.chunking import Chunk
from telco_support_agent.generation import RAGPipeline, format_sources
from telco_support_agent.golden import GoldenItem
from telco_support_agent.judge import Judge, Verdict
from telco_support_agent.vector_index import SearchResult

RESULTS_DIR = Path("data/eval/results")
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class ItemResult(BaseModel):
    # from the golden set
    id: str
    category: str
    question: str
    market: str
    answerable: bool
    reference: str
    # from generation
    status: str
    answer: str
    cited: list[str]  # chunk ids the answer cites
    retrieved: list[str]  # chunk ids the generator saw, best first
    checks_ok: bool  # lesson 04's deterministic verify(): citations valid, numbers supported
    cost_usd: float
    latency_s: float
    # from judging (None until judged; None also means "not applicable")
    verdict: Verdict | None = None
    verdict_by: str | None = None  # "rule" (deterministic) or the judge model
    reasoning: str = ""
    faithful: bool | None = None  # only for answered responses
    unsupported_claims: list[str] = []
    citation_relevant: bool | None = None  # a cited chunk is relevant per the golden labels
    judge_cost_usd: float = 0.0


class GenerationReport(BaseModel):
    created: str
    label: str
    config: dict[str, str | int]
    summary: dict[str, float | None]  # None = not applicable (e.g. no answered responses)
    by_category: dict[str, dict[str, float | None]]
    items: list[ItemResult]

    def save(self, directory: Path = RESULTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"generation_{self.label}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def load(cls, label: str, directory: Path = RESULTS_DIR) -> "GenerationReport":
        return cls(**json.loads((directory / f"generation_{label}.json").read_text()))


def parallel[T, R](fn: Callable[[T], R], items: list[T], workers: int = 8) -> list[R]:
    """Map `fn` over `items` on a thread pool, keeping order. LLM calls are network-bound, so
    threads give a near-linear speed-up even with Python's GIL."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def generate(items: list[GoldenItem], pipeline: RAGPipeline, workers: int = 8) -> list[ItemResult]:
    # Embed every question once, up front: afterwards the threads only READ the embedding cache
    # (writing it from several threads at once is not safe).
    pipeline.retriever.embedder.embed([i.question for i in items])

    def run(item: GoldenItem) -> ItemResult:
        r = pipeline.ask(item.question, item.market)
        return ItemResult(
            id=item.id,
            category=item.category,
            question=item.question,
            market=item.market,
            answerable=item.answerable,
            reference=item.reference_answer,
            status=r.status,
            answer=r.answer,
            cited=[c.chunk_id for c in r.citations],
            retrieved=[chunk_id for chunk_id, _ in r.retrieved],
            checks_ok=r.checks.ok,
            cost_usd=r.cost_usd,
            latency_s=r.latency_s,
        )

    return parallel(run, items, workers)


def judge_item(
    result: ItemResult, item: GoldenItem, judge: Judge, chunks: dict[str, Chunk]
) -> ItemResult:
    """Fill in the judgment fields. Refusals are graded by rule; answers by the LLM judge."""
    r = result.model_copy(update={"unsupported_claims": [], "judge_cost_usd": 0.0})
    answered = r.status == "answered"

    if not r.answerable or not answered:
        # No judge needed: an unanswerable question must be refused; an answerable one must not.
        r.verdict = "correct" if r.answerable == answered else "incorrect"
        r.verdict_by = "rule"
        r.reasoning = (
            "refused as expected"
            if not r.answerable and not answered
            else "answered an unanswerable question"
            if not r.answerable
            else "refused a question the documents answer"
        )
    else:
        judgment, record = judge.correctness(r.question, r.reference, r.answer)
        r.verdict, r.verdict_by, r.reasoning = judgment.verdict, judge.model, judgment.reasoning
        r.judge_cost_usd += record.cost_usd

    if answered:
        results = [
            SearchResult(rank=i + 1, score=0.0, chunk=chunks[cid])
            for i, cid in enumerate(r.retrieved)
        ]
        faith, record = judge.faithfulness(r.answer, format_sources(results))
        r.faithful, r.unsupported_claims = faith.faithful, faith.unsupported_claims
        r.judge_cost_usd += record.cost_usd
        if r.answerable:
            r.citation_relevant = any(item.grade(chunks[cid]) > 0 for cid in r.cited)
    return r


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[ItemResult]) -> dict[str, float | None]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    answered = [r for r in results if r.status == "answered"]
    latencies = [r.latency_s for r in results]
    return {
        "n": len(results),
        # correctness on questions the documents answer (correct = 1, partial = 0.5)
        "correctness": mean([SCORE[r.verdict] for r in answerable if r.verdict]),
        "correct_rate": mean([r.verdict == "correct" for r in answerable]),
        "partial_rate": mean([r.verdict == "partial" for r in answerable]),
        "incorrect_rate": mean([r.verdict == "incorrect" for r in answerable]),
        # refusals
        "refusal_accuracy": mean([r.status == "cannot_answer" for r in unanswerable]),
        "over_refusal": mean([r.status == "cannot_answer" for r in answerable]),
        # grounding (answered responses only)
        "faithfulness": mean([bool(r.faithful) for r in answered if r.faithful is not None]),
        "checks_ok": mean([r.checks_ok for r in answered]),
        "citation_relevance": mean(
            [bool(r.citation_relevant) for r in answered if r.citation_relevant is not None]
        ),
        # cost and speed
        "cost_per_1k_usd": 1000 * mean([r.cost_usd for r in results]),
        "judge_cost_usd": sum(r.judge_cost_usd for r in results),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
    }


def by_category(results: list[ItemResult]) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
    return {
        cat: {
            "n": len(rs),
            "score": mean([SCORE[r.verdict] for r in rs if r.verdict]),
            "faithful": mean([bool(r.faithful) for r in rs if r.faithful is not None]),
        }
        for cat, rs in sorted(groups.items())
    }


def judge_all(
    results: list[ItemResult],
    items: list[GoldenItem],
    judge: Judge,
    chunks: list[Chunk],
    workers: int = 8,
) -> list[ItemResult]:
    by_id = {i.id: i for i in items}
    chunk_map = {c.chunk_id: c for c in chunks}
    return parallel(lambda r: judge_item(r, by_id[r.id], judge, chunk_map), results, workers)


def build_report(
    results: list[ItemResult], label: str, config: dict[str, str | int]
) -> GenerationReport:
    return GenerationReport(
        created=datetime.now().isoformat(timespec="seconds"),
        label=label,
        config=config,
        summary=summarize(results),
        by_category=by_category(results),
        items=results,
    )
