"""A brute-force vector index in numpy. See learn/03-embeddings-vector-search.md.

Exact search over every vector: for a few thousand chunks it takes well under a millisecond, and
it's the ground truth that approximate indexes (HNSW, in Chroma or Qdrant) are measured against.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from telco_support_agent.chunking import Chunk

INDEX_DIR = Path("data/index")


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector (each row, for a matrix) to length 1. Zero vectors stay zero."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """The `k` rows of `matrix` most similar to `query`, by cosine similarity.

    - `query` has shape (d,), `matrix` has shape (n, d). Neither is assumed normalized.
    - Returns up to `k` (row_index, score) pairs, highest score first; fewer if n < k.
    - Ties keep the lower row index first.
    - Raises ValueError if k < 1.
    """
    if k < 1:
        raise ValueError("k should be >= 1")
    query_norm = normalize(query)
    matrix_norm = normalize(matrix)

    cosine_sim = matrix_norm @ query_norm
    top_k = np.argsort(cosine_sim, descending=True)[:k]
    return [(int(i), float(cosine_sim[i])) for i in top_k]


class SearchResult(BaseModel):
    rank: int  # 1 = best
    score: float  # cosine similarity, -1..1
    chunk: Chunk


def _matches(chunk: Chunk, where: dict[str, Any]) -> bool:
    """Exact-match metadata filter. A list value means "any of these"."""
    for field, wanted in where.items():
        value = getattr(chunk, field)
        if isinstance(wanted, list | tuple | set):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model: str):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self.model = model

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self, query_vector: np.ndarray, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Top-k chunks for a query vector, optionally pre-filtered on chunk metadata.

        Example: where={"market": ["KSA", "ALL"], "superseded_by": None}
        """
        rows = [i for i, c in enumerate(self.chunks) if not where or _matches(c, where)]
        if not rows:
            return []
        hits = cosine_top_k(query_vector, self.vectors[rows], k)
        return [
            SearchResult(rank=r, score=round(score, 4), chunk=self.chunks[rows[i]])
            for r, (i, score) in enumerate(hits, start=1)
        ]

    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        meta = {
            "model": self.model,
            "count": len(self),
            "dims": int(self.vectors.shape[1]),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "VectorIndex":
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"no index in {directory}; run scripts/build_index.py first")
        meta = json.loads((directory / "meta.json").read_text())
        lines = (directory / "chunks.jsonl").read_text().splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines]
        return cls(chunks, np.load(directory / "vectors.npy"), meta["model"])
