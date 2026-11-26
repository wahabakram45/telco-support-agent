"""How far can we trust the LLM judge? Compare its verdicts with human labels. See lesson 06.

Flow: make a label sheet from a saved run (answers only, no judge verdicts, so the human isn't
anchored) → a human fills in a verdict per answer → compare with the judge's verdicts.
"""

import hashlib
import random
from collections import Counter  # noqa: F401  (you'll need it)
from pathlib import Path

import yaml
from pydantic import BaseModel

from telco_support_agent.generation_eval import GenerationReport, ItemResult
from telco_support_agent.judge import Verdict

LABELS_PATH = Path("data/eval/human_labels.yaml")
VERDICTS: list[Verdict] = ["correct", "partial", "incorrect"]


def answer_sha(answer: str) -> str:
    return hashlib.sha256(answer.encode()).hexdigest()[:12]


def judged_by_llm(report: GenerationReport) -> list[ItemResult]:
    """Answers the LLM judge graded (refusals are graded by rule, so they need no calibration)."""
    return [r for r in report.items if r.verdict_by not in (None, "rule")]


def sample_for_labelling(report: GenerationReport, n: int = 24, seed: int = 6) -> list[ItemResult]:
    """Enriched sample: every answer the judge didn't call correct, topped up with random correct
    ones, shuffled. Random sampling alone would give ~20 easy "correct"s and almost no signal on
    the hard cases; the price is that agreement on this sample understates agreement overall."""
    rng = random.Random(seed)
    graded = judged_by_llm(report)
    flagged = [r for r in graded if r.verdict != "correct"]
    rest = [r for r in graded if r.verdict == "correct"]
    sample = flagged[:n] + rng.sample(rest, max(0, min(len(rest), n - len(flagged))))
    rng.shuffle(sample)
    return sample


def write_label_sheet(items: list[ItemResult], path: Path = LABELS_PATH) -> None:
    rows = [
        {
            "id": r.id,
            "question": r.question,
            "reference": r.reference,
            "answer": r.answer,
            "answer_sha": answer_sha(r.answer),
            "human": None,
            "note": "",
        }
        for r in items
    ]
    header = (
        "# Human labels for calibrating the LLM judge. Lesson 06, Task 2.\n"
        "# For each answer, replace `human: null` with correct / partial / incorrect, using the\n"
        "# SAME rubric as the judge (learn/06-generation-evals.md §5): judge the answer against\n"
        "# the question and the reference. `note:` is optional: why, in a few words.\n\n"
    )
    path.write_text(header + yaml.safe_dump(rows, sort_keys=False, allow_unicode=True, width=100))


class HumanLabel(BaseModel):
    id: str
    question: str
    reference: str
    answer: str
    answer_sha: str
    human: Verdict | None = None
    note: str = ""


def load_labels(path: Path = LABELS_PATH) -> list[HumanLabel]:
    return [HumanLabel(**row) for row in yaml.safe_load(path.read_text())]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two raters beyond what chance would produce.

    kappa = (p_observed - p_chance) / (1 - p_chance), where p_observed is the share of items
    the raters agree on, and p_chance = sum over labels of (share of a using it) * (share of b
    using it).
    1 = perfect agreement, 0 = no better than chance, < 0 = worse than chance.
    """
    # TODO(you) — lesson 06, Task 1. Raise ValueError for empty or different-length lists.
    # Edge case: if p_chance is 1 (both raters used one and the same label for everything),
    # return 1.0 instead of dividing by zero.
    raise NotImplementedError


def confusion(human: list[str], judge: list[str]) -> dict[str, dict[str, int]]:
    """Rows = human label, columns = judge label."""
    table = {h: dict.fromkeys(VERDICTS, 0) for h in VERDICTS}
    for h, j in zip(human, judge, strict=True):
        table[h][j] += 1
    return table


class Calibration(BaseModel):
    n: int
    agreement: float
    kappa: float
    confusion: dict[str, dict[str, int]]
    disagreements: list[dict[str, str]]
    stale: list[str]  # labelled answers that no longer match the run's answers


def calibrate(labels: list[HumanLabel], report: GenerationReport) -> Calibration:
    by_id = {r.id: r for r in report.items}
    pairs, stale = [], []
    for label in labels:
        if label.human is None:
            continue
        r = by_id.get(label.id)
        if r is None or answer_sha(r.answer) != label.answer_sha:
            stale.append(label.id)
            continue
        pairs.append((label, r))
    if not pairs:
        raise ValueError("no labelled answers match this run")
    human = [label.human for label, _ in pairs]
    judge = [r.verdict for _, r in pairs]
    return Calibration(
        n=len(pairs),
        agreement=sum(h == j for h, j in zip(human, judge, strict=True)) / len(pairs),
        kappa=cohen_kappa(human, judge),
        confusion=confusion(human, judge),
        disagreements=[
            {
                "id": label.id,
                "question": label.question,
                "human": label.human,
                "judge": r.verdict,
                "judge_reasoning": r.reasoning,
                "note": label.note,
            }
            for label, r in pairs
            if label.human != r.verdict
        ],
        stale=stale,
    )
