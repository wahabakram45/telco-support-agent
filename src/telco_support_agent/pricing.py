"""Token cost accounting. See learn/00b-llm-api-fundamentals.md §5."""

from pydantic import BaseModel


class Prices(BaseModel):
    """USD per 1M tokens."""

    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


def estimate_cost(
    input_tokens: int, output_tokens: int, prices: Prices, cached_input_tokens: int = 0
) -> float:
    """Return the USD cost of one call.

    `input_tokens` is the TOTAL input count reported by the API and already INCLUDES
    `cached_input_tokens`. Charge the cached part at the cached price and the rest at the
    normal input price.
    """
    # TODO(you): implement the formula from the lesson. Raise ValueError if
    # cached_input_tokens > input_tokens or any count is negative.
    raise NotImplementedError
