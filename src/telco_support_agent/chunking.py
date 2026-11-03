"""Chunking: split documents into retrievable pieces. See learn/02-chunking.md."""

import re
from collections.abc import Callable
from datetime import date

import tiktoken
from pydantic import BaseModel

from telco_support_agent.documents import Document
from telco_support_agent.llm import count_tokens

_ENC = tiktoken.get_encoding("o200k_base")


class Chunk(BaseModel):
    """One retrievable piece of a document, carrying the document's metadata."""

    chunk_id: str  # "<doc_id>#<n>", n counts from 0 within the document
    doc_id: str
    title: str
    heading: str = ""  # the section heading this chunk came from; "" if none
    text: str  # the chunk's own content
    context: str = ""  # prefix prepended for embedding, e.g. "Roaming Policy (KSA) > GCC Day Pass"
    doc_type: str
    market: str
    plan_ids: list[str] = []
    effective_from: date
    effective_to: date | None = None
    superseded_by: str | None = None

    @property
    def embed_text(self) -> str:
        """What gets embedded and searched: context header + text."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def make_chunk(doc: Document, n: int, text: str, heading: str = "", context: str = "") -> Chunk:
    """Build chunk number `n` of `doc`, copying the document's metadata."""
    meta = doc.model_dump(exclude={"text", "source_path"})
    return Chunk(chunk_id=f"{doc.doc_id}#{n}", heading=heading, text=text, context=context, **meta)


# --- Strategy 1: fixed-size token windows (baseline) ---------------------------------------


def fixed_size_chunks(doc: Document, size: int = 200, overlap: int = 0) -> list[Chunk]:
    """Cut the text into windows of `size` tokens; consecutive windows share `overlap` tokens."""
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got overlap={overlap}, size={size}")
    tokens = _ENC.encode(doc.text)
    chunks, start = [], 0
    while True:
        window = tokens[start : start + size]
        chunks.append(make_chunk(doc, len(chunks), _ENC.decode(window).strip()))
        if start + size >= len(tokens):
            return chunks
        start += size - overlap


# --- Helper for structure-aware chunking ---------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_long(text: str, max_tokens: int) -> list[str]:
    """Split `text` into pieces of at most `max_tokens`, preferring natural boundaries.

    Packs whole paragraphs; a paragraph that is too long is packed sentence by sentence; a
    sentence that is still too long is cut by tokens. The same idea as LangChain's
    RecursiveCharacterTextSplitter.
    """
    units: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append(para)
            continue
        for sentence in _SENTENCE_END.split(para):
            if count_tokens(sentence) <= max_tokens:
                units.append(sentence)
            else:
                toks = _ENC.encode(sentence)
                for i in range(0, len(toks), max_tokens):
                    units.append(_ENC.decode(toks[i : i + max_tokens]))

    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


# --- Strategy 2: structure-aware (markdown headings) ---------------------------------------


def chunk_by_headings(doc: Document, max_tokens: int = 300) -> list[Chunk]:
    """One chunk per markdown section, with a context header. See learn/02-chunking.md, Task 1.

    Rules (tests/test_chunking.py checks them):
    - A section starts at any heading line (`#`, `##` or `###` followed by a space). The text
      before the first `##`/`###` heading belongs to a section with heading "".
    - The `# Title` line itself is not chunk content (the title goes into the context).
    - Sections with no body text are skipped.
    - `chunk.heading` is the heading text without the `#`s; `chunk.text` is the section body.
    - `chunk.context` is "<doc title> > <heading>", or just "<doc title>" when heading is "".
    - A body longer than `max_tokens` is split with `split_long`; every piece keeps the same
      heading and context.
    - Chunks are numbered 0, 1, 2… in document order (use `make_chunk`).
    """
    # TODO(you) — lesson 02, Task 1.
    raise NotImplementedError


# --- Applying a strategy to the corpus ------------------------------------------------------

Strategy = Callable[[Document], list[Chunk]]


def chunk_corpus(docs: list[Document], strategy: Strategy) -> list[Chunk]:
    return [chunk for doc in docs for chunk in strategy(doc)]
