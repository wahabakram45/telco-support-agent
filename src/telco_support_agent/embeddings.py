"""Turn text into vectors with the OpenAI embeddings API. See learn/03-embeddings-vector-search.md.

Every vector is cached on disk by (model, text), so re-running a script, or re-chunking only part of
the corpus, never pays twice for the same text.
"""

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from telco_support_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/embeddings")
# The API accepts up to 2,048 inputs per request; 256 keeps each request far below its token limit.
MAX_BATCH = 256


class EmbeddingStats(BaseModel):
    """Running totals for one Embedder: what was served from cache and what was paid for."""

    texts: int = 0
    cached: int = 0
    api_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()


class Embedder:
    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        cache_dir: Path | None = CACHE_DIR,
    ):
        self.settings = settings or get_settings()
        self.model = self.settings.embedding_model
        self.client = client or OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.request_timeout_s,
            max_retries=self.settings.max_retries,
        )
        self.stats = EmbeddingStats()
        self._cache_path = cache_dir / f"{self.model}.npz" if cache_dir else None
        self._cache: dict[str, np.ndarray] = self._load_cache()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts` in order. Returns a float32 matrix of shape (len(texts), dims)."""
        if any(not t.strip() for t in texts):
            raise ValueError("cannot embed empty text")
        keys = [_key(self.model, t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        self.stats.texts += len(texts)
        self.stats.cached += len(texts) - len(missing)

        pending = list(missing.items())
        for i in range(0, len(pending), MAX_BATCH):
            batch = pending[i : i + MAX_BATCH]
            start = time.perf_counter()
            response = self.client.embeddings.create(
                model=self.model, input=[text for _, text in batch]
            )
            self.stats.latency_s += time.perf_counter() - start
            self.stats.api_calls += 1
            self.stats.tokens += response.usage.total_tokens
            # Results carry an `index`; sort by it rather than trusting the order.
            items = sorted(response.data, key=lambda d: d.index)
            for (key, _), item in zip(batch, items, strict=True):
                self._cache[key] = np.asarray(item.embedding, dtype=np.float32)

        self.stats.cost_usd = self.stats.tokens * self.settings.price_embedding_per_m / 1_000_000
        if missing:
            self._save_cache()
            logger.info("embedded %d new texts: %s", len(missing), self.stats.model_dump())
        return np.stack([self._cache[k] for k in keys]) if keys else np.empty((0, 0), np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self._cache_path or not self._cache_path.exists():
            return {}
        data = np.load(self._cache_path)
        return dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys, vectors = list(self._cache), np.stack(list(self._cache.values()))
        np.savez(self._cache_path, keys=np.array(keys), vectors=vectors)
