"""Cheap metrics for comparing chunking strategies before we have retrieval evals.

The real test of chunking is retrieval quality (Phase 3). These proxies catch obvious problems
early: facts cut in half, and chunks too small or too large to be useful.
"""

from statistics import mean, median

from pydantic import BaseModel

from telco_support_agent.checks import extract_numbers
from telco_support_agent.chunking import Chunk


class FactIntegrity(BaseModel):
    intact: int  # every number of the fact appears together in one chunk
    split: int  # the doc contains the fact's numbers, but no single chunk has all of them
    unchecked: int  # fact has no checkable numbers, or the doc doesn't state them
    split_facts: list[str]  # "<doc_id>: <fact>" for each split fact

    @property
    def rate(self) -> float:
        checked = self.intact + self.split
        return self.intact / checked if checked else 1.0


def fact_integrity(chunks: list[Chunk], facts_by_doc: dict[str, list[str]]) -> FactIntegrity:
    """How many ground-truth facts survive chunking in one piece?

    Uses numbers as a fingerprint: a fact is intact if one chunk of its document contains all
    of the fact's numbers (e.g. "25", "24" for "25 SAR for 1 GB, valid for 24 hours").
    """
    by_doc: dict[str, list[set[str]]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(extract_numbers(c.text))

    intact = split = unchecked = 0
    split_facts = []
    for doc_id, facts in facts_by_doc.items():
        chunk_numbers = by_doc.get(doc_id, [])
        doc_numbers = set().union(*chunk_numbers) if chunk_numbers else set()
        for fact in facts:
            needed = extract_numbers(fact)
            if not needed or not needed <= doc_numbers:
                unchecked += 1
            elif any(needed <= nums for nums in chunk_numbers):
                intact += 1
            else:
                split += 1
                split_facts.append(f"{doc_id}: {fact}")
    return FactIntegrity(intact=intact, split=split, unchecked=unchecked, split_facts=split_facts)


class SizeStats(BaseModel):
    chunks: int
    mean_tokens: float
    median_tokens: float
    max_tokens: int
    tiny: int  # chunks under 30 tokens: often too little context to be useful
    total_tokens: int  # what you'd pay to embed the corpus


def size_stats(chunks: list[Chunk]) -> SizeStats:
    sizes = [c.tokens for c in chunks]
    return SizeStats(
        chunks=len(sizes),
        mean_tokens=round(mean(sizes), 1),
        median_tokens=median(sizes),
        max_tokens=max(sizes),
        tiny=sum(s < 30 for s in sizes),
        total_tokens=sum(sizes),
    )
