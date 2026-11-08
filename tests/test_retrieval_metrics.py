import math

import pytest

from telco_support_agent.retrieval_metrics import (
    bootstrap_ci,
    dcg,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# The worked example from the lesson: 5 results, the 2nd answers the question (grade 2), the 4th
# helps a bit (grade 1). The corpus has 3 relevant chunks in total, grades [2, 2, 1].
GRADES = [0, 2, 0, 1, 0]
IDEAL = [2, 2, 1]


# --- provided ---


def test_hit_at_k():
    assert hit_at_k(GRADES, 5) == 1.0
    assert hit_at_k(GRADES, 1) == 0.0


def test_precision_at_k_divides_by_k():
    assert precision_at_k(GRADES, 5) == pytest.approx(2 / 5)
    assert precision_at_k([2], 5) == pytest.approx(1 / 5)  # fewer results than k


def test_dcg_discounts_by_rank():
    assert dcg([2]) == pytest.approx(3.0)  # (2^2 - 1) / log2(2)
    assert dcg([0, 2]) == pytest.approx(3 / math.log2(3))


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 45 + [0.0] * 5
    low, high = bootstrap_ci(values)
    assert low < 0.9 < high and 0.75 < low and high <= 1.0


# --- your Task 1: recall@k and reciprocal rank ---


def test_recall_at_k():
    assert recall_at_k(GRADES, n_relevant=3, k=5) == pytest.approx(2 / 3)
    assert recall_at_k(GRADES, n_relevant=3, k=2) == pytest.approx(1 / 3)
    assert recall_at_k([0, 0, 0], n_relevant=2, k=3) == 0.0


def test_recall_can_never_reach_1_when_more_relevant_than_k():
    assert recall_at_k([2, 2], n_relevant=4, k=2) == pytest.approx(0.5)


def test_recall_is_undefined_without_relevant_chunks():
    with pytest.raises(ValueError):
        recall_at_k([0, 0], n_relevant=0, k=2)


def test_reciprocal_rank():
    assert reciprocal_rank(GRADES) == pytest.approx(1 / 2)
    assert reciprocal_rank([1, 0, 2]) == 1.0  # partially relevant still counts
    assert reciprocal_rank([0, 0, 0, 0, 2]) == pytest.approx(1 / 5)
    assert reciprocal_rank([0, 0]) == 0.0
    assert reciprocal_rank([]) == 0.0


# --- your Task 2: nDCG@k ---


def test_ndcg_perfect_ranking_is_1():
    assert ndcg_at_k([2, 2, 1, 0, 0], IDEAL, 5) == pytest.approx(1.0)


def test_ndcg_worked_example():
    # DCG = 3/log2(3) + 1/log2(5); ideal DCG = 3/1 + 3/log2(3) + 1/2
    expected = (3 / math.log2(3) + 1 / math.log2(5)) / (3 + 3 / math.log2(3) + 0.5)
    assert ndcg_at_k(GRADES, IDEAL, 5) == pytest.approx(expected)
    assert ndcg_at_k(GRADES, IDEAL, 5) == pytest.approx(0.4308, abs=1e-4)


def test_ndcg_rewards_putting_the_best_chunk_first():
    assert ndcg_at_k([2, 1], [2, 1], 2) > ndcg_at_k([1, 2], [2, 1], 2)


def test_ndcg_ideal_is_cut_at_k():
    # Only 1 slot: the ideal is the single best chunk, so a grade-2 result at rank 1 is perfect.
    assert ndcg_at_k([2, 0, 0], IDEAL, 1) == pytest.approx(1.0)


def test_ndcg_is_undefined_without_relevant_chunks():
    with pytest.raises(ValueError):
        ndcg_at_k([0, 0], [], 2)
