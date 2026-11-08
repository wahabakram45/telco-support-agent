"""The golden set: fixed questions with known relevant chunks. See learn/05-retrieval-evals.md.

Relevance is labelled by *content*, not by chunk ID: "a chunk of document D that contains the text
T". Chunk IDs (`doc#3`) change whenever the chunking changes; documents and facts don't, so the
labels survive re-chunking and re-wording.
"""

from functools import partial
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from telco_support_agent.chunking import Chunk, chunk_by_headings, chunk_corpus
from telco_support_agent.documents import load_documents
from telco_support_agent.generation import Market

GOLDEN_PATH = Path("data/eval/golden.yaml")

Category = Literal[
    "single_fact",  # one chunk answers it, worded close to the document
    "paraphrase",  # one chunk answers it, worded the way a customer would (little word overlap)
    "multi_doc",  # needs chunks from two or more documents
    "near_duplicate",  # T1: Flex Smart vs Flex Smart Plus
    "regional",  # T2: same plan name, other market
    "price_history",  # T3: a 2025 version exists with an older price
    "unlimited_fair_usage",  # T4: "unlimited" + the fair usage policy
    "discontinued",  # T5: the plan can't be bought any more
    "superseded_policy",  # T6: the 2025 roaming policy
    "market_rule",  # T7: a rule that differs by market (VAT)
    "buried_fact",  # T8: one sentence deep in the long T&C
    "multilingual",  # Arabic or mixed-language questions
    "out_of_scope",  # no document answers it
]


def normalize(text: str) -> str:
    """Lowercase, drop markdown bold, straighten apostrophes, collapse whitespace."""
    return " ".join(text.replace("*", "").replace("\u2019", "'").lower().split())


class Relevant(BaseModel):
    """Which chunks count as relevant: chunks of `doc` whose text contains `has`."""

    model_config = ConfigDict(extra="forbid")

    doc: str
    has: str | None = None  # ignores case and **bold**; None = every chunk of the document
    grade: Literal[1, 2] = 2  # 2 = answers the question, 1 = partial or useful context

    def matches(self, chunk: Chunk) -> bool:
        if chunk.doc_id != self.doc:
            return False
        return self.has is None or normalize(self.has) in normalize(chunk.text)


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    market: Market
    category: Category
    source: Literal["hand", "synthetic", "seed"]
    relevant: list[Relevant] = []  # empty = no document answers it
    distractors: list[str] = []  # doc_ids that should NOT be retrieved (wrong plan, discontinued…)
    reference_answer: str = ""  # what a correct answer says; used by the generation evals (06)
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)

    def grade(self, chunk: Chunk) -> int:
        """The chunk's relevance grade for this question: the highest matching label, else 0."""
        return max((r.grade for r in self.relevant if r.matches(chunk)), default=0)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    items = [GoldenItem(**row) for row in yaml.safe_load(path.read_text())]
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate golden ids: {sorted(duplicates)}")
    return items


def load_corpus_chunks() -> list[Chunk]:
    """The chunks the index is built from (same strategy as scripts/build_index.py)."""
    return chunk_corpus(load_documents(), partial(chunk_by_headings, max_tokens=300))
