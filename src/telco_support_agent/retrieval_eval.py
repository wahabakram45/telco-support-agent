"""Run the golden set through the retriever and score it. See learn/05-retrieval-evals.md.

Run: uv run python scripts/eval_retrieval.py
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from telco_support_agent.chunking import Chunk
from telco_support_agent.generation import market_filter
from telco_support_agent.golden import GoldenItem
from telco_support_agent.retrieval import Retriever
from telco_support_agent.retrieval_metrics import (
    bootstrap_ci,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from telco_support_agent.vector_index import SearchResult

RESULTS_DIR = Path("data/eval/results")
METRICS = ["hit", "recall", "precision", "mrr", "ndcg", "distractor"]


class Retrieved(BaseModel):
    chunk_id: str
    score: float
    grade: int
    distractor: bool


class ItemResult(BaseModel):
    id: str
    category: str
    question: str
    market: str
    n_relevant: int  # relevant chunks in the corpus for this question
    retrieved: list[Retrieved]
    hit: float
    recall: float
    precision: float
    mrr: float
    ndcg: float
    distractor: float  # 1.0 if a distractor document made the top k


class RetrievalReport(BaseModel):
    created: str
    label: str
    k: int
    config: dict[str, str | int]
    n_items: int
    n_scored: int
    skipped: list[str]  # unanswerable items: retrieval metrics don't apply
    overall: dict[str, float]
    ci95: dict[str, tuple[float, float]]
    hit_curve: dict[int, float]  # hit@1, @3, @5, @10: how deep you must go
    by_category: dict[str, dict[str, float]]
    items: list[ItemResult]

    def save(self, directory: Path = RESULTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"retrieval_{self.label}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path


def score_item(
    item: GoldenItem, results: list[SearchResult], corpus: list[Chunk], k: int
) -> ItemResult:
    """Score one question from its ranked results. `corpus` gives the denominators."""
    ideal = [g for c in corpus if (g := item.grade(c)) > 0]
    if not ideal:
        raise ValueError(f"{item.id}: no chunk in the corpus matches its relevance labels")
    grades = [item.grade(r.chunk) for r in results]
    retrieved = [
        Retrieved(
            chunk_id=r.chunk.chunk_id,
            score=round(r.score, 4),
            grade=g,
            distractor=r.chunk.doc_id in item.distractors,
        )
        for r, g in zip(results, grades, strict=True)
    ]
    return ItemResult(
        id=item.id,
        category=item.category,
        question=item.question,
        market=item.market,
        n_relevant=len(ideal),
        retrieved=retrieved,
        hit=hit_at_k(grades, k),
        recall=recall_at_k(grades, len(ideal), k),
        precision=precision_at_k(grades, k),
        mrr=reciprocal_rank(grades[:k]),
        ndcg=ndcg_at_k(grades, ideal, k),
        distractor=float(any(r.distractor for r in retrieved[:k])),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def evaluate(
    items: list[GoldenItem],
    retriever: Retriever,
    corpus: list[Chunk],
    k: int = 5,
    label: str = "dev",
    depth: int = 10,
) -> RetrievalReport:
    """Retrieve `depth` results per answerable question (for the hit curve); score the top k."""
    scored: list[ItemResult] = []
    hit_curve: dict[int, list[float]] = {n: [] for n in (1, 3, 5, depth)}
    for item in items:
        if not item.answerable:
            continue
        results = retriever.retrieve(item.question, k=depth, where=market_filter(item.market))
        scored.append(score_item(item, results, corpus, k))
        grades = [g.grade for g in scored[-1].retrieved]
        for n in hit_curve:
            hit_curve[n].append(hit_at_k(grades, n))

    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for r in scored:
        groups[r.category].append(r)

    golden_bytes = json.dumps([i.model_dump() for i in items], sort_keys=True).encode()
    return RetrievalReport(
        created=datetime.now().isoformat(timespec="seconds"),
        label=label,
        k=k,
        config={
            "embedding_model": retriever.embedder.model,
            "index_chunks": len(retriever.index),
            "corpus_chunks": len(corpus),
            "golden_sha": hashlib.sha256(golden_bytes).hexdigest()[:12],
        },
        n_items=len(items),
        n_scored=len(scored),
        skipped=[i.id for i in items if not i.answerable],
        overall={m: mean([getattr(r, m) for r in scored]) for m in METRICS},
        ci95={m: bootstrap_ci([getattr(r, m) for r in scored]) for m in METRICS},
        hit_curve={n: mean(v) for n, v in hit_curve.items()},
        by_category={
            cat: {"n": len(rs), **{m: mean([getattr(r, m) for r in rs]) for m in METRICS}}
            for cat, rs in sorted(groups.items())
        },
        items=scored,
    )
