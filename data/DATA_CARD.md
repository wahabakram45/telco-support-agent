# Data Card — TelcoX Support Corpus

Synthetic dataset for a fictional mobile operator, **TelcoX**, operating in two markets: **KSA (SAR)**
and **UAE (AED)**. Built in [Lesson 01](../learn/01-data-for-rag.md). Contains no real company or
customer data.

## Contents

| Part | Path | Size | Source of truth? |
|---|---|---|---|
| Plan catalog | `catalog/plans.json` | 37 plan versions (32 current: 16 per market) | **Yes**, for prices and allowances. Tools read this |
| Plan pages | `docs/plans/*.md` | 37 docs, one per plan version | No: rendered from `plans.json`, so they can't drift |
| Policies | `docs/policy/*.md` | 21 docs | No: generated from the facts in `scripts/generate_docs.py` |
| FAQs | `docs/faq/*.md` | 4 docs | No: generated from facts, in customer-style wording |

Total: **62 documents, ~9,000 words**. By market: 25 KSA, 22 UAE, 15 apply to both (`ALL`).

Current plans per market: 5 prepaid (Flex), 4 postpaid (Max), 3 data-only (DataGo), 4 roaming add-ons (Travel).

## Metadata (YAML frontmatter on every document)

`doc_id`, `title`, `doc_type` (plan / policy / faq), `market` (KSA / UAE / ALL), `plan_ids`,
`effective_from`, `effective_to` (null = current), `superseded_by`.

## Ground truth

- **Plans:** `plans.json`.
- **Policies and FAQs:** the `facts` list of each topic in `scripts/generate_docs.py`. Every number a
  document states must come from its facts; `checks.find_unsupported_numbers` enforced this during
  generation (all 25 passed on the first attempt).

## Deliberate traps (each becomes an eval category in Phase 3)

| Trap | Where | What a bot gets wrong |
|---|---|---|
| T1 Near-duplicate names | Flex Smart vs Flex Smart Plus (both markets) | Mixes up the two plans' price or allowance |
| T2 Regional variants | Every plan name exists in KSA and UAE at different prices | Quotes the other market's price or currency |
| T3 Price history | Flex Smart (KSA), Max Advanced (KSA), DataGo 60 (UAE): 2025 versions ended 2026-06-30 | Quotes the outdated price |
| T4 Unlimited + fair usage | Flex Unlimited, Max Premium, Max Elite (both markets) + `fair-usage-policy` | Says "no limits", or treats the threshold as the allowance. A full answer needs two documents |
| T5 Discontinued, no replacement | Flex Classic (KSA), Social Pack (UAE) | Recommends a plan that can't be bought |
| T6 Superseded policy | `roaming-policy-ksa-2025` vs `roaming-policy-ksa` | Quotes 2025 roaming rates (40 vs 30 SAR / 100 MB) |
| T7 Market-specific rules | `postpaid-billing-ksa` (VAT 15% included) vs `postpaid-billing-uae` (VAT 5% added) | Applies one market's VAT rule to the other |
| T8 Buried fact | Senior discount (65+, 20% postpaid) in the ~1,450-word `terms-and-conditions` | Lost in the middle, or split across chunks |

## Regenerating

```bash
uv run python scripts/generate_plans.py           # catalog + plan pages (deterministic)
uv run python scripts/generate_docs.py --force    # policies + FAQs (LLM, ~$0.015, wording varies per run)
```

Plan pages are deterministic. Policy and FAQ **wording** changes on every regeneration, but the facts
don't. Regenerating changes the corpus, so re-run the evals afterwards.

## Known limitations

- Clean markdown only: no PDFs, tables in images or OCR, unlike real enterprise corpora.
- English only.
- LLM-written prose sounds more uniform than real help pages.
- The T&C repeats some facts in a recap section, so the buried fact appears twice.
