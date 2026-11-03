from datetime import date

import pytest

from telco_support_agent.chunk_metrics import fact_integrity, size_stats
from telco_support_agent.chunking import (
    chunk_by_headings,
    fixed_size_chunks,
    make_chunk,
    split_long,
)
from telco_support_agent.documents import Document
from telco_support_agent.llm import count_tokens

POLICY = """# Roaming Policy (KSA)

This page is for KSA customers.

## GCC Day Pass
The pass costs 25 SAR for 1 GB, valid for 24 hours.

## Activation
Dial *123# or use the app."""

FAQ = """# Roaming FAQs

### How do I turn on roaming?
Use the app or dial *123#.

### How much is the day pass?
25 SAR in KSA."""


def doc(text: str, doc_id: str = "roaming-policy-ksa", title: str = "Roaming Policy (KSA)"):
    return Document(
        doc_id=doc_id,
        title=title,
        doc_type="policy",
        market="KSA",
        effective_from=date(2026, 7, 1),
        source_path="x.md",
        text=text,
    )


# --- fixed-size baseline (provided) ---


def test_fixed_size_covers_all_tokens_without_overlap():
    d = doc(POLICY)
    chunks = fixed_size_chunks(d, size=10)
    assert sum(count_tokens(c.text) for c in chunks) >= count_tokens(POLICY) - len(chunks)
    assert [c.chunk_id for c in chunks] == [f"roaming-policy-ksa#{i}" for i in range(len(chunks))]


def test_fixed_size_overlap_makes_more_chunks():
    d = doc(POLICY)
    assert len(fixed_size_chunks(d, 10, overlap=5)) > len(fixed_size_chunks(d, 10))


def test_fixed_size_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed_size_chunks(doc(POLICY), size=10, overlap=10)


def test_chunks_carry_document_metadata():
    c = make_chunk(doc(POLICY), 0, "text")
    assert (c.market, c.doc_type, c.effective_from) == ("KSA", "policy", date(2026, 7, 1))


def test_embed_text_prepends_context():
    c = make_chunk(doc(POLICY), 0, "body", context="Title > Heading")
    assert c.embed_text == "Title > Heading\n\nbody"
    assert make_chunk(doc(POLICY), 0, "body").embed_text == "body"


def test_split_long_respects_limit_and_keeps_text():
    text = "\n\n".join(f"Paragraph {i} has a few words in it." for i in range(30))
    pieces = split_long(text, max_tokens=40)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 40 for p in pieces)
    assert " ".join(pieces).split() == text.split()


# --- structure-aware chunker (your Task 1) ---


def test_headings_one_chunk_per_section():
    chunks = chunk_by_headings(doc(POLICY))
    assert [c.heading for c in chunks] == ["", "GCC Day Pass", "Activation"]
    assert [c.text for c in chunks] == [
        "This page is for KSA customers.",
        "The pass costs 25 SAR for 1 GB, valid for 24 hours.",
        "Dial *123# or use the app.",
    ]
    assert [c.chunk_id for c in chunks] == [f"roaming-policy-ksa#{i}" for i in range(3)]


def test_headings_context_header():
    chunks = chunk_by_headings(doc(POLICY))
    assert chunks[0].context == "Roaming Policy (KSA)"
    assert chunks[1].context == "Roaming Policy (KSA) > GCC Day Pass"


def test_headings_no_heading_lines_in_text():
    for c in chunk_by_headings(doc(POLICY)) + chunk_by_headings(doc(FAQ, "faq", "Roaming FAQs")):
        assert not any(line.startswith("#") for line in c.text.splitlines())


def test_headings_faq_questions_become_chunks():
    chunks = chunk_by_headings(doc(FAQ, "faq-roaming", "Roaming FAQs"))
    assert [c.heading for c in chunks] == ["How do I turn on roaming?", "How much is the day pass?"]


def test_headings_doc_without_subheadings_is_one_chunk():
    chunks = chunk_by_headings(doc("# Flex Smart (KSA)\n\nPrepaid plan.\n\n**Price:** 55 SAR"))
    assert len(chunks) == 1 and chunks[0].heading == ""
    assert chunks[0].text == "Prepaid plan.\n\n**Price:** 55 SAR"


def test_headings_long_section_is_split_with_same_heading():
    body = "\n\n".join(f"Clause {i} says something about the service." for i in range(40))
    chunks = chunk_by_headings(doc(f"# T&C\n\n## General\n{body}", title="T&C"), max_tokens=50)
    assert len(chunks) > 1
    assert all(c.heading == "General" and c.context == "T&C > General" for c in chunks)
    assert all(c.tokens <= 50 for c in chunks)


# --- metrics (provided) ---


def test_fact_integrity_detects_split_facts():
    d = doc(POLICY)
    facts = {"roaming-policy-ksa": ["The GCC Day Pass costs 25 SAR for 1 GB, valid for 24 hours."]}
    whole = [make_chunk(d, 0, "costs 25 SAR for 1 GB, valid for 24 hours")]
    halves = [make_chunk(d, 0, "costs 25 SAR for 1 GB,"), make_chunk(d, 1, "valid for 24 hours")]
    assert fact_integrity(whole, facts).rate == 1.0
    report = fact_integrity(halves, facts)
    assert report.split == 1 and report.rate == 0.0


def test_size_stats():
    d = doc(POLICY)
    s = size_stats([make_chunk(d, 0, "short"), make_chunk(d, 1, "word " * 50)])
    assert s.chunks == 2 and s.tiny == 1 and s.max_tokens >= 50
