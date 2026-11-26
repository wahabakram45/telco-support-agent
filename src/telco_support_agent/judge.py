"""LLM-as-judge for generated answers. See learn/06-generation-evals.md.

Two separate judges, because they compare the answer against different things:
- correctness: the answer vs the golden reference answer ("is it right?");
- faithfulness: the answer vs the sources the generator saw ("is every claim supported?").
A right answer can be unfaithful (guessed correctly), and a faithful answer can be wrong (the
sources were the wrong ones). Each judge returns its reasoning BEFORE its verdict.
"""

from typing import Literal

from pydantic import BaseModel, Field

from telco_support_agent.config import judge_settings
from telco_support_agent.llm import CallRecord, LLMClient

Verdict = Literal["correct", "partial", "incorrect"]

CORRECTNESS_PROMPT = """You grade a telecom support assistant's answer against a reference answer.

Verdicts:
- "correct": states the facts the question needs, consistent with the reference. Extra true
  details and different wording are fine. Missing a minor aside is fine.
- "partial": right in part, but misses something the customer needs (e.g. the price but not the
  validity when both were asked, or one of two options), or is too vague to act on.
- "incorrect": a key fact contradicts the reference (wrong price, wrong plan, wrong market or
  currency, a discontinued plan offered as available), or it refuses when the reference answers.

Rules:
- Judge facts, not style or length. Numbers must match exactly (25 SAR is not 20 SAR).
- The reference may be shorter than a good answer; don't penalize correct extra detail.
- If the answer is in another language, judge its meaning.
Write your reasoning first, then the verdict."""

FAITHFULNESS_PROMPT = """You check whether a support assistant's answer is supported by its sources.

List every factual claim in the answer that the sources do NOT support: invented numbers, plans,
steps, timelines, conditions, or a fact taken from a source for the wrong plan or market.
Paraphrase is fine. Polite filler ("contact support if you need help") is not a claim. Common
knowledge used only to interpret the question (Bahrain is in the GCC) is fine.
The answer is faithful if the list is empty. Write your reasoning first."""


class CorrectnessJudgment(BaseModel):
    reasoning: str = Field(description="Compare the answer's key facts with the reference.")
    verdict: Verdict


class FaithfulnessJudgment(BaseModel):
    reasoning: str = Field(description="Check each claim in the answer against the sources.")
    unsupported_claims: list[str] = Field(description="Claims the sources don't support.")

    @property
    def faithful(self) -> bool:
        return not self.unsupported_claims


class Judge:
    def __init__(self, llm: LLMClient | None = None, correctness_prompt: str | None = None):
        self.llm = llm or LLMClient(judge_settings())
        self.correctness_prompt = correctness_prompt or CORRECTNESS_PROMPT

    @property
    def model(self) -> str:
        return self.llm.settings.llm_model

    def correctness(
        self, question: str, reference: str, answer: str
    ) -> tuple[CorrectnessJudgment, CallRecord]:
        user = (
            f"<question>{question}</question>\n"
            f"<reference_answer>{reference}</reference_answer>\n"
            f"<answer>{answer}</answer>"
        )
        return self.llm.extract(self.correctness_prompt, user, CorrectnessJudgment)

    def faithfulness(self, answer: str, sources: str) -> tuple[FaithfulnessJudgment, CallRecord]:
        user = f"<sources>\n{sources}\n</sources>\n\n<answer>{answer}</answer>"
        return self.llm.extract(FAITHFULNESS_PROMPT, user, FaithfulnessJudgment)
