from datetime import date

import pytest
from pydantic import ValidationError

from telco_support_agent.catalog import Plan

BASE = dict(
    plan_id="flex-50-ksa",
    name="Flex 50",
    market="KSA",
    currency="SAR",
    plan_type="prepaid",
    price=50,
    validity_days=28,
    data_gb=10,
    local_minutes=300,
    sms=100,
    effective_from="2026-01-01",
)


def plan(**overrides) -> Plan:
    return Plan(**{**BASE, **overrides})


def test_valid_plan_with_defaults():
    p = plan()
    assert p.effective_from == date(2026, 1, 1)
    assert p.unlimited_data is False and p.fair_usage_gb is None
    assert p.roaming_countries == [] and p.features == []
    assert p.status == "active" and p.replaced_by is None and p.effective_to is None
    assert p.is_current


def test_unlimited_plan():
    p = plan(data_gb=None, unlimited_data=True, fair_usage_gb=100)
    assert p.unlimited_data and p.data_gb is None


def test_roaming_addon_with_countries():
    p = plan(plan_type="roaming_addon", roaming_countries=["UAE"], roaming_data_gb=5)
    assert p.roaming_countries == ["UAE"]


def test_superseded_version_is_not_current():
    p = plan(effective_to="2026-06-30", replaced_by="flex-50-ksa-v2")
    assert not p.is_current


def test_json_round_trip():
    p = plan()
    assert Plan.model_validate_json(p.model_dump_json()) == p


@pytest.mark.parametrize(
    "overrides",
    [
        {"currency": "AED"},  # currency must match market
        {"market": "Oman"},
        {"plan_type": "family"},
        {"price": 0},
        {"price": -5},
        {"validity_days": 0},
        {"local_minutes": -1},
        {"sms": -1},
        {"data_gb": -1},
        {"plan_id": "Flex 50"},  # not a slug
        {"status": "paused"},
        {"colour": "blue"},  # unknown field
        {"unlimited_data": True, "data_gb": None},  # unlimited needs fair_usage_gb
        {"unlimited_data": True, "fair_usage_gb": 100},  # unlimited must have data_gb=None
        {"data_gb": None},  # data_gb None only when unlimited
        {"plan_type": "roaming_addon"},  # roaming add-on needs countries
        {"effective_to": "2025-12-31"},  # must be after effective_from
        {"effective_to": "2026-01-01"},
        {"replaced_by": "flex-50-ksa-v2"},  # replaced_by needs effective_to
    ],
)
def test_rejects_invalid_plans(overrides):
    with pytest.raises(ValidationError):
        plan(**overrides)
