from datetime import date

import pytest

from telco_support_agent.documents import (
    DOCS_ROOT,
    load_documents,
    parse_markdown,
    to_markdown,
)


def test_markdown_round_trip():
    meta = {"doc_id": "x", "effective_from": date(2026, 1, 1), "effective_to": None}
    body = "# Title\n\nText with a horizontal rule:\n\n---\n\nmore text"
    parsed_meta, parsed_body = parse_markdown(to_markdown(meta, body))
    assert parsed_meta == meta
    assert parsed_body == body


def test_missing_frontmatter_rejected():
    with pytest.raises(ValueError):
        parse_markdown("# no frontmatter")


def _write(root, name, doc_id, superseded_by=None):
    meta = {
        "doc_id": doc_id,
        "title": doc_id,
        "doc_type": "policy",
        "market": "ALL",
        "effective_from": date(2026, 1, 1),
        "superseded_by": superseded_by,
    }
    (root / name).write_text(to_markdown(meta, "# body"))


def test_duplicate_ids_rejected(tmp_path):
    _write(tmp_path, "a.md", "same")
    _write(tmp_path, "b.md", "same")
    with pytest.raises(ValueError, match="duplicate"):
        load_documents(tmp_path)


def test_broken_superseded_link_rejected(tmp_path):
    _write(tmp_path, "a.md", "old", superseded_by="missing")
    with pytest.raises(ValueError, match="superseded_by"):
        load_documents(tmp_path)


@pytest.mark.skipif(not DOCS_ROOT.exists(), reason="generate the corpus first")
def test_real_corpus_loads():
    docs = load_documents()
    assert len(docs) >= 50
    assert {d.doc_type for d in docs} == {"plan", "policy", "faq"}
    assert any(not d.is_current for d in docs), "corpus should include superseded versions"
