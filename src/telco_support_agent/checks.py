"""Cheap, deterministic checks on generated text. A first taste of faithfulness evals."""

import re

# Digits, optional thousands groups (",000"), optional decimal part (".99").
# A "%" after the number is simply not captured, so "15%" yields "15".
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _normalize(raw: str) -> str:
    """'1,000' -> '1000', '50.0' -> '50', '49.90' -> '49.9'."""
    n = raw.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def extract_numbers(text: str) -> set[str]:
    """All normalized numbers in `text`, skipping single-digit integers."""
    found = {_normalize(m) for m in _NUMBER.findall(text)}
    return {n for n in found if len(n) > 1}


def find_unsupported_numbers(text: str, source: str) -> set[str]:
    """Return numbers that appear in `text` but nowhere in `source`.

    Normalization (applied to both sides):
    - thousands separators are removed: "1,000" == "1000"
    - trailing zero decimals are dropped: "50.0" == "50", but "49.99" stays "49.99"
    - a number followed by "%" counts as the number: "15%" -> "15"
    Single-digit integers (0–9) are ignored: they're mostly list numbering and generic words.
    """
    return extract_numbers(text) - extract_numbers(source)
