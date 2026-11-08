"""Retrieval metrics. See learn/05-retrieval-evals.md.

Every function takes `grades`: the relevance grade of each retrieved chunk, in rank order
(0 = not relevant, 1 = partially relevant, 2 = answers the question). Example: [0, 2, 0, 1, 0]
means the 2nd result answers the question and the 4th helps a bit.
"""

import math

import numpy as np


def hit_at_k(grades: list[int], k: int) -> float:
    """1.0 if any of the top k is relevant, else 0.0: did the LLM get at least one useful chunk?"""
    return 1.0 if any(g > 0 for g in grades[:k]) else 0.0


def precision_at_k(grades: list[int], k: int) -> float:
    """Share of the top k that is relevant. Divides by k even if fewer results came back."""
    return sum(1 for g in grades[:k] if g > 0) / k


def recall_at_k(grades: list[int], n_relevant: int, k: int) -> float:
    """Share of ALL the relevant chunks in the corpus that made it into the top k.

    `n_relevant` counts every relevant chunk in the corpus, retrieved or not.
    """
    # TODO(you) — lesson 05, Task 1. Raise ValueError if n_relevant is 0.
    raise NotImplementedError


def reciprocal_rank(grades: list[int]) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none is relevant. Ranks start at 1."""
    # TODO(you) — lesson 05, Task 1.
    raise NotImplementedError


def dcg(grades: list[int]) -> float:
    """Discounted cumulative gain: each result's gain (2^grade - 1), divided by log2(rank + 1)."""
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))


def ndcg_at_k(grades: list[int], ideal_grades: list[int], k: int) -> float:
    """DCG of the top k, divided by the best possible DCG of k results (1.0 = perfect order).

    `ideal_grades` are the grades of ALL relevant chunks in the corpus (any order): the ideal
    ranking puts the highest grades first.
    """
    # TODO(you) — lesson 05, Task 2. Use dcg(). Raise ValueError if the ideal DCG is 0.
    raise NotImplementedError


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Confidence interval for the mean, by resampling the questions with replacement."""
    if not values:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = rng.choice(data, size=(n_resamples, len(data)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)
