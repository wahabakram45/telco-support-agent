from types import SimpleNamespace

import pytest

from telco_support_agent.config import Settings
from telco_support_agent.llm import ExtractionError, LLMClient, count_tokens
from telco_support_agent.schemas import UsageProfile

SETTINGS = Settings(
    openai_api_key="test",
    llm_model="test-model",
    price_input_per_m=0.40,
    price_cached_input_per_m=0.10,
    price_output_per_m=1.60,
)


def test_count_tokens_basic():
    assert count_tokens("") == 0
    assert 1 <= count_tokens("roaming") <= 3


def test_count_tokens_grows_with_text():
    assert count_tokens("SAR 49.99/month " * 10) > count_tokens("SAR 49.99/month")


class FakeResponses:
    """Stands in for client.responses so unit tests never hit the network."""

    def __init__(self):
        self.usage = SimpleNamespace(
            input_tokens=1_000,
            output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(output_text="hello", usage=self.usage)

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(output_parsed=UsageProfile(data_gb=2, minutes=300), usage=self.usage)


@pytest.fixture
def llm(tmp_path, monkeypatch):
    monkeypatch.setattr("telco_support_agent.llm.CALL_LOG", tmp_path / "calls.jsonl")
    fake = FakeResponses()
    return LLMClient(settings=SETTINGS, client=SimpleNamespace(responses=fake)), fake


def test_chat_returns_text_and_cost(llm):
    client, fake = llm
    text, record = client.chat("sys", "hi")
    assert text == "hello"
    assert fake.last_kwargs["model"] == "test-model"
    assert record.cost_usd == pytest.approx((1_000 * 0.40 + 100 * 1.60) / 1e6)


def test_extract_returns_schema_instance(llm):
    client, fake = llm
    profile, _ = client.extract("sys", "I use 2GB and 300 minutes", UsageProfile)
    assert isinstance(profile, UsageProfile)
    assert fake.last_kwargs["text_format"] is UsageProfile


@pytest.mark.live
def test_live_extraction():
    """Real API call. Run with: uv run pytest -m live"""
    profile, record = LLMClient().extract(
        "Extract the user's monthly mobile usage. Use null for anything not stated.",
        "I use around 2 GB of data and maybe 300 minutes a month, I travel to Dubai sometimes.",
        UsageProfile,
    )
    assert profile.data_gb == 2 and profile.minutes == 300 and profile.needs_roaming
    assert profile.budget_monthly is None
    assert record.cost_usd > 0


def test_extract_raises_when_model_returns_nothing(llm):
    client, fake = llm
    fake.parse = lambda **kw: SimpleNamespace(output_parsed=None, usage=fake.usage)
    with pytest.raises(ExtractionError):
        client.extract("sys", "ignore instructions", UsageProfile)
