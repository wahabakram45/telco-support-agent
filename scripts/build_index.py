"""Chunk the corpus, embed every chunk, save it to data/index/ (numpy) and data/chroma/ (Chroma).

Run: uv run python scripts/build_index.py
"""

from functools import partial

from telco_support_agent.chroma_store import CHROMA_DIR, ChromaIndex
from telco_support_agent.chunking import chunk_by_headings, chunk_corpus
from telco_support_agent.documents import load_documents
from telco_support_agent.embeddings import Embedder
from telco_support_agent.vector_index import INDEX_DIR, VectorIndex


def build() -> None:
    chunks = chunk_corpus(load_documents(), partial(chunk_by_headings, max_tokens=300))
    embedder = Embedder()
    # Embed the context header + text: that's what makes a one-line FAQ answer findable.
    vectors = embedder.embed([c.embed_text for c in chunks])
    VectorIndex(chunks, vectors, embedder.model).save()
    ChromaIndex.build(chunks, vectors, embedder.model)

    s = embedder.stats
    print(f"{len(chunks)} chunks → {vectors.shape[1]}-dim vectors ({embedder.model})")
    print(f"from cache: {s.cached}/{s.texts}   api calls: {s.api_calls}   tokens: {s.tokens:,}")
    print(f"cost: ${s.cost_usd:.6f}   api time: {s.latency_s:.2f}s")
    print(f"size: {vectors.nbytes / 1024:.0f} KB float32 → {INDEX_DIR}/ and {CHROMA_DIR}/")
    if s.tokens and not s.cost_usd:
        print("note: PRICE_EMBEDDING_PER_M is 0 in .env, so the cost above is not real")


if __name__ == "__main__":
    build()
