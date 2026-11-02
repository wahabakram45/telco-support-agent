"""Thin wrapper around the OpenAI Responses API that logs tokens and cost for every call.

See learn/00b-llm-api-fundamentals.md §7 for why this exists.
"""

import json
import logging
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from telco_support_agent.config import Settings, get_settings
from telco_support_agent.pricing import Prices, estimate_cost

logger = logging.getLogger(__name__)

CALL_LOG = Path("logs/llm_calls.jsonl")


class CallRecord(BaseModel):
    """One row per LLM call. Phase 6 aggregates these into cost per 1k queries and p95 latency."""

    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens locally, before sending anything.

    o200k_base is the encoding used by current OpenAI models.
    """
    # TODO(you): use tiktoken.get_encoding(...) and encode the text.
    raise NotImplementedError


class LLMClient:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        self.settings = settings or get_settings()
        # The SDK retries 429/5xx with exponential backoff itself; we just configure it.
        self.client = client or OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.request_timeout_s,
            max_retries=self.settings.max_retries,
        )
        self.prices = Prices(
            input_per_m=self.settings.price_input_per_m,
            cached_input_per_m=self.settings.price_cached_input_per_m,
            output_per_m=self.settings.price_output_per_m,
        )

    def _record(self, usage, latency_s: float) -> CallRecord:
        """Turn the API's `response.usage` into a CallRecord and append it to the call log."""
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        record = CallRecord(
            model=self.settings.llm_model,
            input_tokens=usage.input_tokens,
            cached_input_tokens=cached,
            output_tokens=usage.output_tokens,
            cost_usd=estimate_cost(usage.input_tokens, usage.output_tokens, self.prices, cached),
            latency_s=round(latency_s, 3),
        )
        CALL_LOG.parent.mkdir(exist_ok=True)
        with CALL_LOG.open("a") as f:
            f.write(json.dumps(record.model_dump()) + "\n")
        logger.info("llm call %s", record.model_dump())
        return record

    def chat(
        self, system: str, user: str, temperature: float = 0.0, max_output_tokens: int = 500
    ) -> tuple[str, CallRecord]:
        """Plain text completion. Returns (text, record)."""
        start = time.perf_counter()  # noqa: F841 (used in your TODO; delete this noqa then)
        # TODO(you): call self.client.responses.create(...) with
        #   model=self.settings.llm_model, instructions=system, input=user,
        #   temperature=temperature, max_output_tokens=max_output_tokens
        # then return (response.output_text, self._record(response.usage, elapsed)).
        raise NotImplementedError

    def extract[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> tuple[T, CallRecord]:
        """Structured output: returns a validated instance of `schema`."""
        start = time.perf_counter()  # noqa: F841 (used in your TODO; delete this noqa then)
        # TODO(you): call self.client.responses.parse(..., text_format=schema, temperature=0)
        # and return (response.output_parsed, self._record(...)).
        # What if output_parsed is None (e.g. the model refused)? Decide and handle it.
        raise NotImplementedError
