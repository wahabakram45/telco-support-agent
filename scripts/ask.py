"""Ask the RAG pipeline one question.

Run: uv run python scripts/ask.py "how much is the GCC day pass?" --market KSA
"""

import argparse

from telco_support_agent.generation import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--market", choices=["KSA", "UAE"], default="KSA")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sources", action="store_true", help="also list all retrieved chunks")
    args = parser.parse_args()

    r = RAGPipeline(k=args.k).ask(args.question, args.market)
    print(f"[{r.status}]\n{r.answer}\n")
    for c in r.citations:
        print(f"  {c.label} → {c.chunk_id}  ({c.title} > {c.heading or '-'})")
    if args.sources:
        print("\nretrieved:")
        for chunk_id, score in r.retrieved:
            print(f"  {score:.3f}  {chunk_id}")
    print(f"\nchecks: {'ok' if r.checks.ok else r.checks.model_dump()}")
    print(f"cost: ${r.cost_usd:.5f}   latency: {r.latency_s:.1f}s")


if __name__ == "__main__":
    main()
