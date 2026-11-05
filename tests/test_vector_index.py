from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from telco_support_agent.chunking import Chunk
from telco_support_agent.config import Settings
from telco_support_agent.embeddings import Embedder
from telco_support_agent.retrieval import Retriever
from telco_support_agent.vector_index import VectorIndex, cosine_top_k, normalize


def chunk(n: int, market: str = "KSA", superseded_by: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"doc#{n}",
        doc_id="doc",
        title="Doc",
        text=f"chunk {n}",
        doc_type="policy",
        market=market,
        effective_from=date(2026, 7, 1),
        superseded_by=superseded_by,
    )


MATRIX = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])


# --- cosine_top_k (your Task 1) ---


def test_top_k_orders_by_similarity():
    hits = cosine_top_k(np.array([1.0, 0.2]), MATRIX, k=4)
    assert [i for i, _ in hits] == [0, 2, 1, 3]
    assert hits[0][1] == pytest.approx(1 / np.sqrt(1.04))


def test_top_k_returns_python_ints_and_floats():
    i, score = cosine_top_k(np.array([1.0, 0.0]), MATRIX, k=1)[0]
    assert (i, score) == (0, pytest.approx(1.0))
    assert type(i) is int and type(score) is float


def test_top_k_ignores_vector_length():
    # Cosine compares direction only: a longer vector is not "more similar".
    matrix = np.array([[10.0, 0.0], [0.6, 0.8]])
    assert cosine_top_k(np.array([0.6, 0.8]), matrix, k=1)[0][0] == 1
    assert cosine_top_k(np.array([6.0, 8.0]), matrix, k=1)[0][1] == pytest.approx(1.0)


def test_top_k_returns_exactly_k_when_k_is_small():
    # Real vectors have many more dimensions than k: k must not depend on d.
    matrix = np.eye(10)
    assert len(cosine_top_k(np.ones(10), matrix, k=3)) == 3


def test_top_k_larger_than_n_returns_all():
    assert len(cosine_top_k(np.array([1.0, 0.0]), MATRIX, k=10)) == 4


def test_top_k_ties_keep_lower_index_first():
    matrix = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    assert [i for i, _ in cosine_top_k(np.array([1.0, 0.0]), matrix, k=2)] == [1, 2]


def test_top_k_rejects_k_below_one():
    with pytest.raises(ValueError):
        cosine_top_k(np.array([1.0, 0.0]), MATRIX, k=0)


# --- index (provided) ---


def test_normalize_makes_unit_rows_and_keeps_zero_rows():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        VectorIndex([chunk(0)], MATRIX, "m")


def test_search_filters_before_ranking():
    chunks = [chunk(0, "KSA"), chunk(1, "UAE"), chunk(2, "ALL"), chunk(3, "KSA", "new-doc")]
    index = VectorIndex(chunks, MATRIX, "m")
    results = index.search(np.array([1.0, 0.0]), k=5, where={"market": ["UAE", "ALL"]})
    assert [r.chunk.chunk_id for r in results] == ["doc#2", "doc#1"]
    assert [r.rank for r in results] == [1, 2]
    current = index.search(np.array([-1.0, 0.0]), k=1, where={"superseded_by": None})
    assert current[0].chunk.chunk_id != "doc#3"
    assert index.search(np.array([1.0, 0.0]), where={"market": "XX"}) == []


def test_save_and_load_round_trip(tmp_path):
    index = VectorIndex([chunk(i) for i in range(4)], MATRIX, "m")
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.model == "m" and len(loaded) == 4
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.chunks == index.chunks


# --- embedder with a fake API (provided) ---


class FakeEmbeddings:
    """Embeds text as [len(text), 1.0]. Returns results in reverse order to test re-sorting."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input):
        self.calls.append(input)
        data = [SimpleNamespace(index=i, embedding=[len(t), 1.0]) for i, t in enumerate(input)]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(total_tokens=10))


def fake_embedder(tmp_path) -> tuple[Embedder, FakeEmbeddings]:
    fake = FakeEmbeddings()
    settings = Settings(openai_api_key="x", price_embedding_per_m=0.02)
    client = SimpleNamespace(embeddings=fake)
    return Embedder(settings, client, cache_dir=tmp_path), fake


def test_embedder_keeps_input_order(tmp_path):
    embedder, _ = fake_embedder(tmp_path)
    vectors = embedder.embed(["a", "bbb", "cc"])
    assert vectors[:, 0].tolist() == [1, 3, 2]


def test_embedder_cache_avoids_second_call(tmp_path):
    embedder, fake = fake_embedder(tmp_path)
    embedder.embed(["a", "bb"])
    embedder.embed(["bb", "a", "ccc"])
    assert fake.calls == [["a", "bb"], ["ccc"]]
    assert embedder.stats.cached == 2 and embedder.stats.cost_usd == pytest.approx(20 * 0.02e-6)
    reloaded, fake2 = fake_embedder(tmp_path)  # the cache survives on disk
    reloaded.embed(["a"])
    assert fake2.calls == []


def test_embedder_rejects_empty_text(tmp_path):
    embedder, _ = fake_embedder(tmp_path)
    with pytest.raises(ValueError):
        embedder.embed(["ok", "  "])


def test_retriever_refuses_mismatched_models(tmp_path):
    embedder, _ = fake_embedder(tmp_path)
    index = VectorIndex([chunk(0)], np.array([[1.0, 0.0]]), "some-other-model")
    with pytest.raises(ValueError, match="rebuild"):
        Retriever(index, embedder)
