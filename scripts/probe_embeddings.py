"""Poke at embeddings to see what they capture and what they miss. See lesson 03, Task 2.

Run: uv run python scripts/probe_embeddings.py pairs     # sentence-pair similarities
     uv run python scripts/probe_embeddings.py queries   # probe queries against the index
"""

import sys

import numpy as np

from telco_support_agent.embeddings import Embedder
from telco_support_agent.retrieval import Retriever
from telco_support_agent.vector_index import normalize

# (label, sentence A, sentence B). Predict each score before running.
PAIRS = [
    ("paraphrase", "How do I turn on roaming?", "How can I use mobile data when I travel abroad?"),
    ("same topic", "How do I turn on roaming?", "GCC roaming costs 30 SAR per 100 MB."),
    ("unrelated", "How do I turn on roaming?", "My favourite food is chicken biryani."),
    ("negation", "I want to cancel my plan.", "I do not want to cancel my plan."),
    ("numbers", "The plan costs 55 SAR.", "The plan costs 95 SAR."),
    ("market", "Roaming prices for KSA customers", "Roaming prices for UAE customers"),
    ("arabic", "How do I turn on roaming?", "كيف أقوم بتفعيل التجوال؟"),
    ("typo", "How do I turn on roaming?", "how do i trun on romaing"),
]

# (what it tests, query). Predict whether the right chunk comes first.
QUERIES = [
    ("synonyms: no shared words with the doc", "using my phone internet while on holiday in Dubai"),
    ("version trap T7: old and new price", "how much is the GCC day pass"),
    ("buried fact T8 in a 2,000-word T&C", "do elderly customers get a discount"),
    ("exact identifier", "flex-smart-2025-ksa"),
    ("arabic query", "كم سعر باقة التجوال الخليجية؟"),
    ("out of domain: there is no answer", "what is the weather in Riyadh tomorrow"),
]


def pairs() -> None:
    embedder = Embedder()
    texts = [t for _, a, b in PAIRS for t in (a, b)]
    vectors = normalize(embedder.embed(texts))
    for i, (label, a, b) in enumerate(PAIRS):
        score = float(vectors[2 * i] @ vectors[2 * i + 1])
        print(f"{score:.3f}  {label:11} {a!r}  vs  {b!r}")
    dims = vectors.shape[1]
    norms = np.linalg.norm(embedder.embed(texts[:1]), axis=1)
    print(f"\n{dims} dimensions; raw vector length {norms[0]:.4f} (OpenAI returns unit vectors)")
    print(f"cost so far: ${embedder.stats.cost_usd:.6f} ({embedder.stats.tokens} tokens)")


def queries() -> None:
    retriever = Retriever()
    for label, query in QUERIES:
        print(f"=== {label}\n    {query!r}")
        for r in retriever.retrieve(query, k=3):
            print(f"    {r.rank}. {r.score:.3f}  {r.chunk.chunk_id:32} {r.chunk.heading[:40]}")
        print()


if __name__ == "__main__":
    {"pairs": pairs, "queries": queries}[sys.argv[1] if len(sys.argv) > 1 else "pairs"]()
