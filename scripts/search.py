"""Search the index from the command line.

Run: uv run python scripts/search.py "how much is roaming in dubai"
     uv run python scripts/search.py "gcc day pass price" --market KSA --current --k 3
"""

import argparse

from telco_support_agent.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--market", choices=["KSA", "UAE"], help="keep this market + ALL")
    parser.add_argument("--current", action="store_true", help="drop superseded documents")
    args = parser.parse_args()

    where = {}
    if args.market:
        where["market"] = [args.market, "ALL"]
    if args.current:
        where["superseded_by"] = None

    for r in Retriever().retrieve(args.question, k=args.k, where=where or None):
        c = r.chunk
        preview = " ".join(c.text.split())[:160]
        print(f"{r.rank}. {r.score:.3f}  {c.chunk_id}  [{c.market}, from {c.effective_from}]")
        print(f"   {c.context}\n   {preview}\n")


if __name__ == "__main__":
    main()
