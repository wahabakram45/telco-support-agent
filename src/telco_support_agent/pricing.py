"""Token cost accounting. See learn/00b-llm-api-fundamentals.md §5."""

from pydantic import BaseModel, Field


class Prices(BaseModel):
    """USD per 1M tokens."""

    input_per_m: float = Field(ge=0)
    cached_input_per_m: float = Field(ge=0)
    output_per_m: float = Field(ge=0)


def estimate_cost(
    input_tokens: int, output_tokens: int, prices: Prices, cached_input_tokens: int = 0
) -> float:
    """Return the USD cost of one call.

    `input_tokens` is the TOTAL input count reported by the API and already INCLUDES
    `cached_input_tokens`. Charge the cached part at the cached price and the rest at the
    normal input price.
    """
    if min(input_tokens, output_tokens, cached_input_tokens) < 0:
        raise ValueError(
            f"Token counts can't be negative: input={input_tokens}, "
            f"output={output_tokens}, cached={cached_input_tokens}"
        )
    if cached_input_tokens > input_tokens:
        raise ValueError(
            f"cached_input_tokens ({cached_input_tokens}) > input_tokens ({input_tokens})"
        )

    mil = 1_000_000
    input_cost = ((input_tokens - cached_input_tokens) / mil) * prices.input_per_m
    output_cost = (output_tokens / mil) * prices.output_per_m
    cached_input_cost = (cached_input_tokens / mil) * prices.cached_input_per_m
    return input_cost + output_cost + cached_input_cost
