"""Generation: answer from retrieved chunks, with citations, or say it can't. See lesson 04.

Flow: retrieve (market-filtered, current only) → number the sources S1..Sk → the LLM answers as
structured output → deterministic checks (citations exist, numbers appear in the sources).
"""

import json
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from telco_support_agent.checks import find_unsupported_numbers
from telco_support_agent.llm import LLMClient
from telco_support_agent.retrieval import Retriever
from telco_support_agent.vector_index import SearchResult

ANSWER_LOG = Path("logs/rag_answers.jsonl")
Market = Literal["KSA", "UAE"]

# TODO(you) — lesson 04, Task 1: replace this naive prompt with a grounded one (see the lesson).
SYSTEM_PROMPT = """You are a helpful assistant. Answer the customer's question."""


class Answer(BaseModel):
    """What the LLM must return (structured output)."""

    status: Literal["answered", "cannot_answer"]
    answer: str = Field(description="The reply to the customer.")
    citations: list[str] = Field(description='Labels of the sources used, e.g. ["S1", "S3"].')


class Citation(BaseModel):
    label: str
    chunk_id: str
    title: str
    heading: str


class Checks(BaseModel):
    """Deterministic checks on one answer: a first, cheap layer of faithfulness evaluation."""

    invalid_citations: list[str] = []  # labels that were never given to the model
    missing_citations: bool = False  # status is "answered" but nothing valid is cited
    unsupported_numbers: list[str] = []  # numbers in the answer found in no cited source

    @property
    def ok(self) -> bool:
        return not (self.invalid_citations or self.missing_citations or self.unsupported_numbers)


class RAGResponse(BaseModel):
    question: str
    market: Market
    status: Literal["answered", "cannot_answer"]
    answer: str
    citations: list[Citation]
    retrieved: list[tuple[str, float]]  # (chunk_id, score), best first
    checks: Checks
    cost_usd: float
    latency_s: float


_CITATION = re.compile(r"\[(S\d+)\]")


def label(i: int) -> str:
    return f"S{i + 1}"


def format_sources(results: list[SearchResult]) -> str:
    """Number the chunks S1..Sk, best first, with the metadata the model needs to judge them."""
    blocks = []
    for i, r in enumerate(results):
        c = r.chunk
        blocks.append(
            f'<source id="{label(i)}" title="{c.title}" section="{c.heading}" '
            f'market="{c.market}" effective_from="{c.effective_from}">\n{c.text}\n</source>'
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[SearchResult], market: Market) -> str:
    # Static rules live in the system prompt (cacheable); per-request data goes here.
    # The question comes last, after the sources: models attend best to the end of the prompt.
    sources = format_sources(results) if results else "(no sources found)"
    return (
        f"Customer's market: {market}\n\n"
        f"<sources>\n{sources}\n</sources>\n\n"
        f"Customer's question: {question}"
    )


def verify(
    answer: Answer, results: list[SearchResult], question: str = ""
) -> tuple[Checks, list[str]]:
    """Check citations and numbers. Returns (checks, valid labels in order of first use)."""
    given = {label(i): r for i, r in enumerate(results)}
    used = list(dict.fromkeys(_CITATION.findall(answer.answer) + answer.citations))
    valid = [lab for lab in used if lab in given]

    # Numbers must appear in the sources the answer cites (all sources, if it cites none), with
    # their metadata (dates), or in the question itself: echoing "50%" back to deny it is fine.
    cited = [given[lab] for lab in valid] or results
    allowed = format_sources(cited) + "\n" + question
    answer_text = _CITATION.sub("", answer.answer)
    checks = Checks(
        invalid_citations=[lab for lab in used if lab not in given],
        missing_citations=answer.status == "answered" and not valid,
        unsupported_numbers=sorted(find_unsupported_numbers(answer_text, allowed)),
    )
    return checks, valid


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        k: int = 5,
        system_prompt: str | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()
        self.k = k
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def ask(self, question: str, market: Market) -> RAGResponse:
        start = time.perf_counter()
        # The market comes from the customer's account, not from the question: a hard filter.
        where = {"market": [market, "ALL"], "superseded_by": None}
        results = self.retriever.retrieve(question, k=self.k, where=where)

        user = build_user_prompt(question, results, market)
        answer, record = self.llm.extract(self.system_prompt, user, Answer)
        checks, valid = verify(answer, results, question)

        by_label = {label(i): r.chunk for i, r in enumerate(results)}
        response = RAGResponse(
            question=question,
            market=market,
            status=answer.status,
            answer=answer.answer,
            citations=[
                Citation(
                    label=lab,
                    chunk_id=by_label[lab].chunk_id,
                    title=by_label[lab].title,
                    heading=by_label[lab].heading,
                )
                for lab in valid
            ],
            retrieved=[(r.chunk.chunk_id, r.score) for r in results],
            checks=checks,
            cost_usd=record.cost_usd,  # generation only; the query embedding is ~1000x cheaper
            latency_s=round(time.perf_counter() - start, 3),
        )
        ANSWER_LOG.parent.mkdir(exist_ok=True)
        with ANSWER_LOG.open("a") as f:
            f.write(json.dumps(response.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return response
