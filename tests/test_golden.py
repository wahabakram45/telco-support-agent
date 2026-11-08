"""The golden set is test data, so it gets tests too: one wrong label skews every metric."""

from datetime import date

import pytest

from telco_support_agent.chunking import Chunk
from telco_support_agent.golden import GoldenItem, Relevant, load_corpus_chunks, load_golden
from telco_support_agent.retrieval_eval import score_item
from telco_support_agent.vector_index import SearchResult


def chunk(doc: str, n: int, text: str, market: str = "KSA") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{n}",
        doc_id=doc,
        title=doc,
        text=text,
        doc_type="policy",
        market=market,
        effective_from=date(2026, 7, 1),
    )


# --- relevance labels ---


def test_label_matches_by_document_and_content():
    label = Relevant(doc="roaming", has="25 sar per 100 mb")
    assert label.matches(chunk("roaming", 1, "GCC: **25 SAR per 100 MB**"))  # case and bold ignored
    assert not label.matches(chunk("roaming", 2, "Other countries: 50 SAR per 100 MB"))
    assert not label.matches(chunk("faq", 1, "GCC: 25 SAR per 100 MB"))  # wrong document


def test_label_without_text_matches_the_whole_document():
    assert Relevant(doc="plan-x").matches(chunk("plan-x", 0, "anything"))


def test_grade_is_the_best_matching_label():
    item = GoldenItem(
        id="t",
        question="?",
        market="KSA",
        category="single_fact",
        source="hand",
        relevant=[Relevant(doc="d", has="price", grade=1), Relevant(doc="d", has="25 SAR")],
    )
    assert item.grade(chunk("d", 0, "The price is 25 SAR")) == 2
    assert item.grade(chunk("d", 1, "The price changed")) == 1
    assert item.grade(chunk("e", 0, "The price is 25 SAR")) == 0


def test_score_item_end_to_end():
    corpus = [chunk("d", 0, "25 SAR"), chunk("d", 1, "also 25 SAR"), chunk("x", 0, "noise")]
    item = GoldenItem(
        id="t",
        question="?",
        market="KSA",
        category="single_fact",
        source="hand",
        relevant=[Relevant(doc="d", has="25 SAR")],
        distractors=["x"],
    )
    results = [SearchResult(rank=i + 1, score=0.5, chunk=c) for i, c in enumerate(corpus[::-1])]
    r = score_item(item, results, corpus, k=2)  # ranked: x#0, d#1, d#0
    assert (r.n_relevant, r.hit, r.recall, r.mrr, r.distractor) == (2, 1.0, 0.5, 0.5, 1.0)


# --- the real golden set ---

ITEMS = load_golden()
CORPUS = load_corpus_chunks()
DOCS = {c.doc_id: c for c in CORPUS}


def test_golden_set_size_and_coverage():
    assert len(ITEMS) >= 75
    assert sum(not i.answerable for i in ITEMS) >= 5  # out-of-scope questions for lesson 06
    assert {i.category for i in ITEMS} >= {"near_duplicate", "regional", "buried_fact"}


@pytest.mark.parametrize("item", [i for i in ITEMS if i.answerable], ids=lambda i: i.id)
def test_every_label_points_at_real_current_chunks_in_the_right_market(item):
    for label in item.relevant:
        assert label.doc in DOCS, f"unknown document {label.doc}"
        assert any(label.matches(c) for c in CORPUS), f"{label} matches no chunk"
        doc = DOCS[label.doc]
        assert doc.market in (item.market, "ALL"), f"{label.doc} is {doc.market}-only"
        assert doc.superseded_by is None, f"{label.doc} is superseded: never retrievable"


@pytest.mark.parametrize("item", [i for i in ITEMS if i.distractors], ids=lambda i: i.id)
def test_distractors_exist(item):
    assert all(d in DOCS for d in item.distractors)
