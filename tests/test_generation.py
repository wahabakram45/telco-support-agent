from datetime import date

import numpy as np

from telco_support_agent import generation
from telco_support_agent.chroma_store import ChromaIndex, to_chroma_where
from telco_support_agent.chunking import Chunk
from telco_support_agent.generation import (
    Answer,
    RAGPipeline,
    build_user_prompt,
    format_sources,
    verify,
)
from telco_support_agent.llm import CallRecord
from telco_support_agent.vector_index import SearchResult, VectorIndex


def chunk(n: int, text: str, market: str = "KSA", superseded_by: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"doc-{n}#0",
        doc_id=f"doc-{n}",
        title=f"Doc {n}",
        heading="Prices",
        text=text,
        doc_type="policy",
        market=market,
        effective_from=date(2026, 7, 1),
        superseded_by=superseded_by,
    )


RESULTS = [
    SearchResult(rank=1, score=0.7, chunk=chunk(1, "The GCC Day Pass costs 25 SAR.")),
    SearchResult(rank=2, score=0.5, chunk=chunk(2, "Billing disputes within 60 days.")),
]


# --- prompt building ---


def test_sources_are_numbered_with_metadata():
    text = format_sources(RESULTS)
    assert '<source id="S1" title="Doc 1" section="Prices" market="KSA"' in text
    assert 'id="S2"' in text and "60 days" in text


def test_question_comes_last_and_empty_retrieval_is_explicit():
    prompt = build_user_prompt("How much?", RESULTS, "KSA")
    assert prompt.startswith("Customer's market: KSA")
    assert prompt.endswith("Customer's question: How much?")
    assert "(no sources found)" in build_user_prompt("How much?", [], "KSA")


# --- verify ---


def test_verify_accepts_a_grounded_answer():
    answer = Answer(status="answered", answer="It costs 25 SAR.", citations=["S1"])
    checks, valid = verify(answer, RESULTS)
    assert checks.ok and valid == ["S1"]


def test_verify_flags_invented_citation_and_missing_citation():
    checks, valid = verify(Answer(status="answered", answer="25 SAR.", citations=["S9"]), RESULTS)
    assert checks.invalid_citations == ["S9"] and checks.missing_citations and valid == []


def test_cannot_answer_needs_no_citation():
    answer = Answer(status="cannot_answer", answer="I don't know.", citations=[])
    checks, _ = verify(answer, RESULTS)
    assert checks.ok


def test_verify_flags_numbers_missing_from_cited_sources():
    # 60 is in S2, but the answer only cites S1: still unsupported.
    answer = Answer(status="answered", answer="25 SAR, disputes in 60 days.", citations=["S1"])
    assert verify(answer, RESULTS)[0].unsupported_numbers == ["60"]


def test_verify_allows_numbers_from_the_question_and_inline_labels():
    answer = Answer(status="answered", answer="Not 40 SAR: it is 25 SAR [S1].", citations=[])
    checks, valid = verify(answer, RESULTS, question="Why is it 40 SAR now?")
    assert checks.ok and valid == ["S1"]


# --- pipeline with fakes ---


class FakeRetriever:
    def __init__(self):
        self.where = None

    def retrieve(self, question, k=5, where=None):
        self.where = where
        return RESULTS[:k]


class FakeLLM:
    def extract(self, system, user, schema):
        answer = schema(status="answered", answer="It costs 25 SAR.", citations=["S1"])
        record = CallRecord(
            model="fake",
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            cost_usd=0.001,
            latency_s=0.1,
        )
        return answer, record


def test_pipeline_filters_by_market_and_maps_citations(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "ANSWER_LOG", tmp_path / "answers.jsonl")
    retriever = FakeRetriever()
    response = RAGPipeline(retriever, FakeLLM(), k=2).ask("How much?", "KSA")
    assert retriever.where == {"market": ["KSA", "ALL"], "superseded_by": None}
    assert response.citations[0].chunk_id == "doc-1#0"
    assert response.retrieved == [("doc-1#0", 0.7), ("doc-2#0", 0.5)]
    assert response.checks.ok
    assert (tmp_path / "answers.jsonl").read_text().count("\n") == 1


# --- Chroma store ---


def test_chroma_where_translation():
    assert to_chroma_where(None) is None
    assert to_chroma_where({"market": "KSA"}) == {"market": {"$eq": "KSA"}}
    assert to_chroma_where({"market": ["KSA", "ALL"], "superseded_by": None}) == {
        "$and": [{"market": {"$in": ["KSA", "ALL"]}}, {"is_current": True}]
    }


def test_chroma_matches_numpy_and_filters(tmp_path):
    chunks = [
        chunk(0, "a"),
        chunk(1, "b", market="UAE"),
        chunk(2, "c", market="ALL"),
        chunk(3, "d", superseded_by="doc-0"),
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.99, 0.01]])
    chroma = ChromaIndex.build(chunks, vectors, "m", path=tmp_path)
    exact = VectorIndex(chunks, vectors, "m")
    query = np.array([1.0, 0.05])

    def ids(index, **kw):
        return [r.chunk.chunk_id for r in index.search(query, k=4, **kw)]

    assert chroma.model == "m" and len(chroma) == 4
    assert ids(chroma) == ids(exact)
    where = {"market": ["KSA", "ALL"], "superseded_by": None}
    assert ids(chroma, where=where) == ["doc-0#0", "doc-2#0"]
    top = chroma.search(query, k=1)[0]
    assert top.score == exact.search(query, k=1)[0].score  # distance converted back
