"""Settings loaded from environment / .env. Never hardcode keys or prices."""

from functools import lru_cache

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
    request_timeout_s: float = 30.0
    max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
