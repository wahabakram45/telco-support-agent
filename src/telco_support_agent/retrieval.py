"""Question in, top-k chunks out: embed the query, search the index."""

from typing import Any

from telco_support_agent.embeddings import Embedder
from telco_support_agent.vector_index import SearchResult, VectorIndex


class Retriever:
    def __init__(self, index: VectorIndex | None = None, embedder: Embedder | None = None):
        self.index = index or VectorIndex.load()
        self.embedder = embedder or Embedder()
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
