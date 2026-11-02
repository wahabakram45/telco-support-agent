"""Smoke test for lesson 00b. Run: uv run python scripts/hello_llm.py"""

import logging

from telco_support_agent.llm import LLMClient, count_tokens
from telco_support_agent.schemas import UsageProfile

logging.basicConfig(level=logging.INFO, format="%(message)s")

llm = LLMClient()

question = "In two sentences: what is mobile roaming and why can it be expensive?"
print(f"prompt tokens (local count): {count_tokens(question)}")
text, rec = llm.chat("You are a concise telecom support assistant.", question)
print(text, "\n", rec, "\n")

for msg in [
    "I use around 2 GB of data and 300 minutes a month.",
    "I use a lot of data, barely call anyone.",  # vague: what gets invented?
    "أستخدم حوالي 2 جيجابايت و300 دقيقة شهريا",  # Arabic: compare token count and extraction
]:
    profile, rec = llm.extract(
        "Extract the user's monthly mobile usage. Use null for anything not explicitly stated.",
        msg,
        UsageProfile,
    )
    print(f"{msg!r}\n  tokens={count_tokens(msg)}  -> {profile}  (${rec.cost_usd:.6f})\n")
