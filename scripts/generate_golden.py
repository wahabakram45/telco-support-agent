"""Draft synthetic golden-set questions from random chunks, for a human to review. Lesson 05.

For each sampled chunk, the LLM writes a question a customer might ask that this chunk answers,
plus a short evidence span copied from the chunk (used as the relevance label). Candidates go to
data/eval/candidates.yaml; nothing reaches golden.yaml without review.

Run: uv run python scripts/generate_golden.py --n 40
"""

import argparse
import random
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from telco_support_agent.chunking import Chunk
from telco_support_agent.golden import load_corpus_chunks
from telco_support_agent.llm import LLMClient

OUT = "data/eval/candidates.yaml"

SYSTEM = """You write test questions for a telecom customer-support search engine.

Given one passage from TelcoX's help documents, write ONE question that a real customer might type
and that this passage answers.

Rules:
- Style "single_fact": a direct question; it may reuse the passage's terms.
- Style "paraphrase": how a customer actually talks. Describe the situation instead of naming the
  policy, and avoid the passage's heading words and key terms (e.g. "my internet crawled after I
  used a lot" instead of "fair usage threshold").
- Never mention "the passage", "the document" or TelcoX's internal terms the customer wouldn't know.
- evidence: copy a SHORT exact span (2–8 words) from the passage that contains the answer, usually
  the number or key phrase. It must appear in the passage character for character.
- answer: one sentence answering the question from the passage only."""


class Candidate(BaseModel):
    question: str
    evidence: str = Field(description="exact 2-8 word span from the passage containing the answer")
    answer: str


def sample(chunks: list[Chunk], n: int, seed: int) -> list[Chunk]:
    """At most one chunk per document, skipping superseded documents and tiny chunks."""
    rng = random.Random(seed)
    eligible = [c for c in chunks if c.superseded_by is None and len(c.text) > 120]
    by_doc: dict[str, list[Chunk]] = {}
    for c in eligible:
        by_doc.setdefault(c.doc_id, []).append(c)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    return [rng.choice(by_doc[d]) for d in docs[:n]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    llm = LLMClient()
    rows, rejected, cost = [], 0, 0.0
    for i, chunk in enumerate(sample(load_corpus_chunks(), args.n, args.seed)):
        style: Literal["single_fact", "paraphrase"] = "paraphrase" if i % 2 else "single_fact"
        market = chunk.market if chunk.market != "ALL" else random.Random(i).choice(["KSA", "UAE"])
        user = f"Style: {style}\n\nPassage ({chunk.title} > {chunk.heading}):\n{chunk.text}"
        candidate, record = llm.extract(SYSTEM, user, Candidate)
        cost += record.cost_usd
        if candidate.evidence.lower() not in chunk.text.lower():
            rejected += 1  # the model paraphrased its "exact" span: unusable as a label
            continue
        rows.append(
            {
                "id": f"syn-{len(rows) + 1:03d}",
                "question": candidate.question,
                "market": market,
                "category": style,
                "source": "synthetic",
                "relevant": [{"doc": chunk.doc_id, "has": candidate.evidence}],
                "reference_answer": candidate.answer,
                "notes": f"generated from {chunk.chunk_id}",
            }
        )
    with open(OUT, "w") as f:
        yaml.safe_dump(rows, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"{len(rows)} candidates → {OUT}   rejected (evidence not in chunk): {rejected}")
    print(f"cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
