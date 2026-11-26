from datetime import date

import pytest

from telco_support_agent.chunking import Chunk
from telco_support_agent.generation_eval import (
    GenerationReport,
    ItemResult,
    build_report,
    judge_item,
    summarize,
)
from telco_support_agent.golden import GoldenItem, Relevant
from telco_support_agent.judge import CorrectnessJudgment, FaithfulnessJudgment
from telco_support_agent.judge_calibration import (
    HumanLabel,
    answer_sha,
    calibrate,
    cohen_kappa,
    confusion,
)
from telco_support_agent.llm import CallRecord

RECORD = CallRecord(
    model="fake",
    input_tokens=1,
    cached_input_tokens=0,
    output_tokens=1,
    cost_usd=0.001,
    latency_s=0.1,
)
CHUNKS = {
    "doc#0": Chunk(
        chunk_id="doc#0",
        doc_id="doc",
        title="Doc",
        text="The GCC Day Pass costs 25 SAR.",
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 7, 1),
    )
}


class FakeJudge:
    model = "fake-judge"

    def __init__(self, verdict="correct", unsupported=()):
        self.verdict, self.unsupported, self.calls = verdict, list(unsupported), []

    def correctness(self, question, reference, answer):
        self.calls.append("correctness")
        return CorrectnessJudgment(reasoning="r", verdict=self.verdict), RECORD

    def faithfulness(self, answer, sources):
        self.calls.append("faithfulness")
        assert "25 SAR" in sources  # the judge sees the retrieved chunks' text
        return FaithfulnessJudgment(reasoning="r", unsupported_claims=self.unsupported), RECORD


def item(answerable: bool = True) -> GoldenItem:
    return GoldenItem(
        id="q",
        question="How much is the day pass?",
        market="KSA",
        category="single_fact" if answerable else "out_of_scope",
        source="hand",
        relevant=[Relevant(doc="doc", has="25 SAR")] if answerable else [],
        reference_answer="25 SAR",
    )


def result(status: str = "answered", answerable: bool = True, **kw) -> ItemResult:
    fields = dict(
        id="q",
        category="single_fact",
        question="How much is the day pass?",
        market="KSA",
        answerable=answerable,
        reference="25 SAR",
        status=status,
        answer="It costs 25 SAR.",
        cited=["doc#0"],
        retrieved=["doc#0"],
        checks_ok=True,
        cost_usd=0.0003,
        latency_s=1.0,
    )
    return ItemResult(**{**fields, **kw})


# --- judging: rules for refusals, the LLM judge for answers ---


def test_answered_question_goes_to_both_judges():
    judge = FakeJudge(verdict="partial", unsupported=["24/7 support"])
    r = judge_item(result(), item(), judge, CHUNKS)
    assert judge.calls == ["correctness", "faithfulness"]
    assert (r.verdict, r.verdict_by, r.faithful) == ("partial", "fake-judge", False)
    assert r.citation_relevant is True and r.judge_cost_usd == pytest.approx(0.002)


@pytest.mark.parametrize(
    ("answerable", "status", "verdict"),
    [
        (False, "cannot_answer", "correct"),  # refused an unanswerable question: right
        (True, "cannot_answer", "incorrect"),  # over-refusal: wrong
    ],
)
def test_refusals_are_graded_by_rule_without_the_llm(answerable, status, verdict):
    judge = FakeJudge()
    r = judge_item(result(status, answerable), item(answerable), judge, CHUNKS)
    assert (r.verdict, r.verdict_by, judge.calls) == (verdict, "rule", [])


def test_answering_an_unanswerable_question_is_incorrect_and_still_checked_for_faithfulness():
    judge = FakeJudge()
    r = judge_item(result("answered", answerable=False), item(False), judge, CHUNKS)
    assert (r.verdict, r.verdict_by, judge.calls) == ("incorrect", "rule", ["faithfulness"])
    assert r.citation_relevant is None


def test_summary():
    results = [
        result(verdict="correct", faithful=True),
        result(verdict="partial", faithful=False),
        result("cannot_answer", verdict="incorrect"),  # over-refusal
        result("cannot_answer", answerable=False, verdict="correct"),
    ]
    s = summarize(results)
    assert s["correctness"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3 answerable
    assert s["refusal_accuracy"] == 1.0 and s["over_refusal"] == pytest.approx(1 / 3)
    assert s["faithfulness"] == pytest.approx(0.5)  # 2 answered: 1 faithful


def test_not_applicable_metrics_survive_a_save_and_load(tmp_path):
    report = build_report([result(verdict="correct", faithful=True)], "t", {})
    assert report.summary["refusal_accuracy"] is None  # no unanswerable questions
    report.save(tmp_path)
    assert GenerationReport.load("t", tmp_path).summary["refusal_accuracy"] is None


# --- calibration ---


def test_confusion_rows_are_human_columns_are_judge():
    table = confusion(["correct", "partial"], ["partial", "partial"])
    assert table["correct"]["partial"] == 1 and table["partial"]["partial"] == 1


def test_calibrate_skips_unlabelled_and_stale_answers():
    report = build_report([result(verdict="partial")], "t", {})
    labels = [
        HumanLabel(
            id="q",
            question="?",
            reference="",
            answer="It costs 25 SAR.",
            answer_sha=answer_sha("It costs 25 SAR."),
            human="correct",
        ),
        HumanLabel(
            id="q",
            question="?",
            reference="",
            answer="old",
            answer_sha=answer_sha("old"),
            human="correct",
        ),
    ]
    c = calibrate(labels, report)
    assert (c.n, c.agreement, c.stale) == (1, 0.0, ["q"])
    assert c.disagreements[0]["judge"] == "partial"


# --- your Task 1: Cohen's kappa ---


def test_kappa_perfect_agreement_is_1():
    labels = ["correct", "partial", "incorrect", "correct"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_worked_example():
    # 10 answers. Human: 6 correct, 4 partial. Judge: 5 correct, 5 partial. They agree on 9.
    human = ["correct"] * 6 + ["partial"] * 4
    judge = ["correct"] * 5 + ["partial"] * 5
    # p_observed = 9/10; p_chance = 0.6*0.5 + 0.4*0.5 = 0.5; kappa = (0.9-0.5)/(1-0.5) = 0.8
    assert cohen_kappa(human, judge) == pytest.approx(0.8)


def test_kappa_punishes_agreement_that_chance_explains():
    # 90% raw agreement looks great, but both raters say "correct" almost always.
    human = ["correct"] * 9 + ["partial"]
    judge = ["correct"] * 10
    assert cohen_kappa(human, judge) == pytest.approx(0.0)


def test_kappa_can_be_negative():
    assert cohen_kappa(["correct", "partial"], ["partial", "correct"]) < 0


def test_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa(["correct"], ["correct", "partial"])
    with pytest.raises(ValueError):
        cohen_kappa([], [])
