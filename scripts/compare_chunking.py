"""Compare chunking strategies on the real corpus.

Run: uv run python scripts/compare_chunking.py
     uv run python scripts/compare_chunking.py --show terms-and-conditions headings-300
"""

import argparse
from functools import partial

from telco_support_agent.chunk_metrics import fact_integrity, size_stats
from telco_support_agent.chunking import chunk_by_headings, chunk_corpus, fixed_size_chunks
from telco_support_agent.documents import load_documents
from telco_support_agent.topics import TOPICS

STRATEGIES = {
    "fixed-100": partial(fixed_size_chunks, size=100),
    "fixed-200": partial(fixed_size_chunks, size=200),
    "fixed-200-overlap-50": partial(fixed_size_chunks, size=200, overlap=50),
    "fixed-500": partial(fixed_size_chunks, size=500),
    "headings-300": partial(chunk_by_headings, max_tokens=300),
}


def compare() -> None:
    docs = load_documents()
    facts = {t.doc_id: t.facts for t in TOPICS}
    print(f"{len(docs)} documents, {len(sum(facts.values(), []))} ground-truth facts\n")
    header = f"{'strategy':22} {'chunks':>6} {'mean':>6} {'max':>5} {'tiny':>5} {'tokens':>7}"
    print(f"{header} {'facts intact':>13}")
    worst = None
    for name, strategy in STRATEGIES.items():
        try:
            chunks = chunk_corpus(docs, strategy)
        except NotImplementedError:
            print(f"{name:22} (not implemented yet)")
            continue
        s, f = size_stats(chunks), fact_integrity(chunks, facts)
        print(
            f"{name:22} {s.chunks:>6} {s.mean_tokens:>6} {s.max_tokens:>5} {s.tiny:>5} "
            f"{s.total_tokens:>7} {f.rate:>12.1%}"
        )
        if worst is None or f.split > worst[1].split:
            worst = (name, f)
    if worst and worst[1].split_facts:
        print(f"\nFacts split by {worst[0]}:")
        for line in worst[1].split_facts:
            print(f"  - {line}")


def show(doc_id: str, strategy_name: str) -> None:
    doc = next(d for d in load_documents() if d.doc_id == doc_id)
    for c in STRATEGIES[strategy_name](doc):
        print(f"--- {c.chunk_id} ({c.tokens} tokens)\n{c.embed_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", nargs=2, metavar=("DOC_ID", "STRATEGY"))
    args = parser.parse_args()
    show(*args.show) if args.show else compare()
