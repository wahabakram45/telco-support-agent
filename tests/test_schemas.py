import pytest
from pydantic import ValidationError

from telco_support_agent.schemas import UsageProfile


def test_all_fields_optional():
    assert UsageProfile().data_gb is None


def test_lax_coercion_from_strings():
    # LLMs sometimes quote numbers; lax mode should accept them
    p = UsageProfile.model_validate({"data_gb": "2.5", "minutes": "300"})
    assert p.data_gb == 2.5 and p.minutes == 300


@pytest.mark.parametrize(
    "field,value",
    [("data_gb", -1), ("minutes", -5), ("sms", -1), ("budget_monthly", -10), ("data_gb", 5000)],
)
def test_rejects_impossible_values(field, value):
    with pytest.raises(ValidationError):
        UsageProfile(**{field: value})


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        UsageProfile(data_gb=2, favourite_colour="blue")


def test_json_round_trip():
    p = UsageProfile(data_gb=2, minutes=300)
    assert UsageProfile.model_validate_json(p.model_dump_json()) == p


def test_every_field_has_a_description_for_the_llm():
    props = UsageProfile.model_json_schema()["properties"]
    assert all("description" in v for v in props.values())
