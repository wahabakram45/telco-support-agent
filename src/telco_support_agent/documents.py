"""The document corpus: markdown files with YAML frontmatter under data/docs/.

See learn/01-data-for-rag.md §2–3.
"""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

DOCS_ROOT = Path("data/docs")

DocType = Literal["plan", "policy", "faq"]
DocMarket = Literal["KSA", "UAE", "ALL"]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    doc_type: DocType
    market: DocMarket
    plan_ids: list[str] = []
    effective_from: date
    effective_to: date | None = None
    superseded_by: str | None = None
    source_path: str
    text: str

    @property
    def is_current(self) -> bool:
        return self.effective_to is None


def to_markdown(meta: dict, body: str) -> str:
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return f"---\n{front}---\n\n{body.strip()}\n"


def parse_markdown(raw: str) -> tuple[dict, str]:
    """Split a file into (frontmatter dict, body)."""
    if not raw.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    _, front, body = raw.split("---\n", 2)
    return yaml.safe_load(front), body.strip()


def load_documents(root: Path = DOCS_ROOT) -> list[Document]:
    """Load and validate every document; fail on duplicate ids or broken superseded_by links."""
    docs = []
    for path in sorted(root.rglob("*.md")):
        meta, body = parse_markdown(path.read_text())
        docs.append(Document(**meta, source_path=str(path), text=body))

    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate doc_ids: {sorted(dupes)}")
    broken = [d.doc_id for d in docs if d.superseded_by and d.superseded_by not in set(ids)]
    if broken:
        raise ValueError(f"superseded_by points to missing docs: {broken}")
    return docs
