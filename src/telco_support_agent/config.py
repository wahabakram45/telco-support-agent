"""Settings loaded from environment / .env. Never hardcode keys or prices."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_model: str = ""
    price_input_per_m: float = 0.0
    price_cached_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    embedding_model: str = "text-embedding-3-small"
    price_embedding_per_m: float = 0.0
    vector_store: Literal["numpy", "chroma"] = "chroma"
    # The LLM judge (lesson 06). Empty = the same model as LLM_MODEL, at the same prices.
    judge_model: str = ""
    price_judge_input_per_m: float = 0.0
    price_judge_cached_input_per_m: float = 0.0
    price_judge_output_per_m: float = 0.0
    request_timeout_s: float = 30.0
    max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


def judge_settings(settings: Settings | None = None) -> Settings:
    """Settings for the judge's LLMClient: the judge model and its prices, if configured."""
    settings = settings or get_settings()
    if not settings.judge_model:
        return settings
    return settings.model_copy(
        update={
            "llm_model": settings.judge_model,
            "price_input_per_m": settings.price_judge_input_per_m,
            "price_cached_input_per_m": settings.price_judge_cached_input_per_m,
            "price_output_per_m": settings.price_judge_output_per_m,
        }
    )
