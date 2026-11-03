"""Generate policy and FAQ documents from fixed facts with the LLM.

Run: uv run python scripts/generate_docs.py [--force] [--only DOC_ID ...]
Writes data/docs/{policy,faq}/*.md. See learn/01-data-for-rag.md Task 4.

Topics and their facts live in telco_support_agent/topics.py.
"""

import argparse

from telco_support_agent.checks import find_unsupported_numbers
from telco_support_agent.documents import DOCS_ROOT, to_markdown
from telco_support_agent.llm import LLMClient
from telco_support_agent.topics import TOPICS, Topic

MAX_ATTEMPTS = 3


MARKET_NOTE = {
    "KSA": "Saudi Arabia (KSA) customers only. Say so near the top.",
    "UAE": "UAE customers only. Say so near the top.",
    "ALL": "All customers in KSA and the UAE. Where a fact differs by market, keep both.",
}

FORMAT_NOTE = {
    "policy": "A policy page: a short intro, then a few sections with descriptive headings. "
    "Prefer short paragraphs; use bullet lists only for real lists.",
    "faq": "An FAQ page: '### ' questions phrased the way real customers ask them (casual, "
    "sometimes vague, e.g. 'Why did my internet get slow?'), each followed by a short answer. "
    "One question can cover several related facts.",
}


def build_prompt(topic: Topic) -> tuple[str, str]:
    """Return (system, user) messages that turn `topic.facts` into a realistic document."""
    system = (
        "You are a technical writer for TelcoX, a mobile operator, writing pages for its "
        "online help centre. Your readers are ordinary customers. Write in clear, plain "
        "English, the way a real operator's help pages read.\n\n"
        "Rules:\n"
        "- Include every fact you are given, with its exact numbers and conditions.\n"
        "- Never add new facts: no extra numbers, prices, fees, deadlines, conditions, "
        "exceptions, phone numbers, URLs or eligibility rules. If something isn't in the "
        "facts, don't mention it.\n"
        "- You may add neutral explanatory text that adds no new rules, e.g. what a term "
        "means or why a customer might care.\n"
        "- Don't number headings or sections. Use bold sparingly, never for whole sentences.\n"
        "- Output markdown only: start with the given '# ' title, no front matter, no "
        "closing remarks to me."
    )
    facts = "\n".join(f"- {f}" for f in topic.facts)
    user = (
        f"Title: # {topic.title}\n"
        f"Market: {MARKET_NOTE[topic.market]}\n"
        f"Length: about {topic.target_words} words. Stay close to this length.\n"
        f"Format: {FORMAT_NOTE[topic.doc_type]}\n"
    )
    if topic.outline:
        per_section = topic.target_words // len(topic.outline)
        headings = "\n".join(f"## {h}" for h in topic.outline)
        user += (
            "This is a long legal-style document. Use exactly these section headings, in this "
            f"order, and write {per_section - 30}-{per_section + 30} words under each:\n"
            f"{headings}\n"
            "Put each fact inside the section where it naturally belongs, as one sentence "
            "among others. Give every fact the same weight: no summaries, no highlights, no "
            "bold, and no extra headings.\n"
        )
    user += f"\nFacts:\n{facts}"
    return system, user


def generate(topic: Topic, llm: LLMClient) -> str:
    """Generate the body, retrying while it contains numbers not supported by the facts."""
    system, user = build_prompt(topic)
    dates = f"effective {topic.effective_from} to {topic.effective_to}"
    source = "\n".join([topic.title, dates, *topic.facts])
    max_tokens = max(800, topic.target_words * 3)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        body, rec = llm.chat(system, user, temperature=0.7, max_output_tokens=max_tokens)
        bad = find_unsupported_numbers(body, source)
        if not bad:
            print(f"  ok (attempt {attempt}, {rec.output_tokens} tokens, ${rec.cost_usd:.5f})")
            return body
        print(f"  attempt {attempt}: unsupported numbers {sorted(bad)}")
    print("  ⚠️  kept the last attempt anyway: review it by hand")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate existing documents")
    parser.add_argument("--only", nargs="*", help="generate only these doc_ids")
    args = parser.parse_args()

    llm = LLMClient()
    for topic in TOPICS:
        if args.only and topic.doc_id not in args.only:
            continue
        path = DOCS_ROOT / topic.doc_type / f"{topic.doc_id}.md"
        if path.exists() and not args.force:
            continue
        print(f"{topic.doc_id}")
        body = generate(topic, llm)
        meta = topic.model_dump(exclude={"facts", "target_words", "outline"})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_markdown(meta, body))


if __name__ == "__main__":
    main()
