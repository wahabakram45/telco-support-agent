"""Checks the generated catalog (data/catalog/plans.json) contains traps T1–T5.

Skipped until you run scripts/generate_plans.py.
"""

from collections import defaultdict

import pytest

from telco_support_agent.catalog import PLANS_PATH, load_plans

pytestmark = pytest.mark.skipif(not PLANS_PATH.exists(), reason="run generate_plans.py first")


@pytest.fixture(scope="module")
def plans():
    return load_plans()


@pytest.fixture(scope="module")
def current(plans):
    return [p for p in plans if p.is_current]


def test_unique_ids(plans):
    ids = [p.plan_id for p in plans]
    assert len(ids) == len(set(ids))


def test_enough_current_plans_in_every_market_and_type(current):
    assert len(current) >= 30
    for market in ("KSA", "UAE"):
        types = {p.plan_type for p in current if p.market == market}
        assert types == {"prepaid", "postpaid", "data_only", "roaming_addon"}, market


def test_t1_near_duplicate_names(current):
    names = {(p.market, p.name) for p in current}
    assert any((m, f"{n} Plus") in names for m, n in names), "need e.g. 'Flex 50' + 'Flex 50 Plus'"


def test_t2_regional_variant(current):
    by_name = defaultdict(dict)
    for p in current:
        by_name[p.name][p.market] = p.price
    assert any(
        len(prices) == 2 and prices["KSA"] != prices["UAE"] for prices in by_name.values()
    ), "need the same plan name in both markets at different prices"


def test_t3_price_history(plans):
    by_id = {p.plan_id: p for p in plans}
    for p in plans:
        if p.replaced_by:
            assert p.replaced_by in by_id, f"{p.plan_id} -> missing {p.replaced_by}"
    histories = [
        (old, by_id[old.replaced_by])
        for old in plans
        if old.replaced_by
        and by_id[old.replaced_by].name == old.name
        and by_id[old.replaced_by].market == old.market
        and by_id[old.replaced_by].price != old.price
        and by_id[old.replaced_by].effective_from > old.effective_to
    ]
    assert len(histories) >= 3, "need ≥3 plans with an older version at a different price"


def test_t4_unlimited_plans(current):
    assert sum(p.unlimited_data for p in current) >= 2


def test_t5_discontinued_without_replacement(plans):
    assert any(p.status == "discontinued" and p.replaced_by is None for p in plans)
