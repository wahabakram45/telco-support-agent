"""The same index in Chroma: a real vector database with an HNSW index. See lesson 04 §6.

Same interface as VectorIndex (`model`, `search`, `__len__`), so Retriever works with either.
We pass our own embeddings, so Chroma never embeds anything itself.
"""

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from telco_support_agent.chunking import Chunk
from telco_support_agent.vector_index import SearchResult

CHROMA_DIR = Path("data/chroma")
COLLECTION = "chunks"


def _metadata(chunk: Chunk) -> dict[str, Any]:
    """Chroma metadata holds only simple values (str/int/float/bool, lists of them); no dates, None
    or dicts. Keep the filter fields simple, and the whole chunk as JSON to rebuild it exactly."""
    return {
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type,
        "market": chunk.market,
        "is_current": chunk.superseded_by is None,
        "chunk_json": chunk.model_dump_json(),
    }


def to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate our filter dict into Chroma's filter language.

    {"market": ["KSA", "ALL"], "superseded_by": None}
      → {"$and": [{"market": {"$in": ["KSA", "ALL"]}}, {"is_current": True}]}
    """
    if not where:
        return None
    clauses = []
    for field, wanted in where.items():
        if field == "superseded_by" and wanted is None:
            clauses.append({"is_current": True})
        elif isinstance(wanted, list | tuple | set):
            clauses.append({field: {"$in": list(wanted)}})
        else:
            clauses.append({field: {"$eq": wanted}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaIndex:
    def __init__(self, path: Path = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_collection(COLLECTION)
        self.model = self.collection.metadata["model"]

    def __len__(self) -> int:
        return self.collection.count()

    @classmethod
    def build(
        cls, chunks: list[Chunk], vectors: np.ndarray, model: str, path: Path = CHROMA_DIR
    ) -> "ChromaIndex":
        """(Re)create the collection from scratch: simplest way to never leave stale chunks."""
        client = chromadb.PersistentClient(path=str(path))
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
        collection = client.create_collection(
            COLLECTION,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"model": model},
        )
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32),
            documents=[c.embed_text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )
        return cls(path)

    def search(
        self, query_vector: np.ndarray, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        result = self.collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32)],
            n_results=k,
            where=to_chroma_where(where),
            include=["distances", "metadatas"],
        )
        # Chroma returns cosine *distance* (1 - similarity); convert back to our score.
        return [
            SearchResult(
                rank=r,
                score=round(1 - distance, 4),
                chunk=Chunk.model_validate_json(meta["chunk_json"]),
            )
            for r, (distance, meta) in enumerate(
                zip(result["distances"][0], result["metadatas"][0], strict=True), start=1
            )
        ]
