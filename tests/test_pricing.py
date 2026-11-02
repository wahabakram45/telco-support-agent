import pytest

from telco_support_agent.pricing import Prices, estimate_cost

PRICES = Prices(input_per_m=0.40, cached_input_per_m=0.10, output_per_m=1.60)


def test_no_cache():
    # 3,000 in + 300 out -> 0.0012 + 0.00048
    assert estimate_cost(3_000, 300, PRICES) == pytest.approx(0.00168)


def test_cached_tokens_are_part_of_input():
    # 2,000 of the 3,000 input tokens hit the cache
    expected = (1_000 * 0.40 + 2_000 * 0.10 + 300 * 1.60) / 1_000_000
    assert estimate_cost(3_000, 300, PRICES, cached_input_tokens=2_000) == pytest.approx(expected)


def test_interview_estimate_per_1k_queries():
    assert 1_000 * estimate_cost(3_000, 300, PRICES) == pytest.approx(1.68)


@pytest.mark.parametrize("args", [(-1, 0, 0), (0, -1, 0), (100, 0, 200)])
def test_rejects_invalid_counts(args):
    inp, out, cached = args
    with pytest.raises(ValueError):
        estimate_cost(inp, out, PRICES, cached_input_tokens=cached)
