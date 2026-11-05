"""Question in, top-k chunks out: embed the query, search the index."""

from typing import Any, Protocol

import numpy as np

from telco_support_agent.embeddings import Embedder
from telco_support_agent.vector_index import SearchResult, VectorIndex


class Index(Protocol):
    """Anything searchable: our numpy VectorIndex, or ChromaIndex."""

    model: str

    def search(
        self, query_vector: np.ndarray, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...


def load_index(store: str) -> Index:
    if store == "chroma":
        from telco_support_agent.chroma_store import ChromaIndex  # heavy import; only if used

        return ChromaIndex()
    return VectorIndex.load()


class Retriever:
    def __init__(self, index: Index | None = None, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index = index or load_index(self.embedder.settings.vector_store)
        # Vectors from different models live in different spaces: comparing them is meaningless.
        if self.index.model != self.embedder.model:
            raise ValueError(
                f"index was built with {self.index.model!r} but queries would use "
                f"{self.embedder.model!r}; rebuild the index"
            )

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        return self.index.search(self.embedder.embed_one(question), k=k, where=where)
