"""Cheap, deterministic checks on generated text. A first taste of faithfulness evals."""


def find_unsupported_numbers(text: str, source: str) -> set[str]:
    """Return numbers that appear in `text` but nowhere in `source`.

    Normalization (applied to both sides):
    - thousands separators are removed: "1,000" == "1000"
    - trailing zero decimals are dropped: "50.0" == "50", but "49.99" stays "49.99"
    - a number followed by "%" counts as the number: "15%" -> "15"
    Single-digit integers (0–9) are ignored: they're mostly list numbering and generic words.
    """
    # TODO(you) — lesson 01, Task 3. Hint: re.findall with a pattern for digits, optional
    # comma groups and an optional decimal part. Write a small normalize() helper.
    raise NotImplementedError
