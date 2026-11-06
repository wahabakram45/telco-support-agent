"""HTTP API around the RAG pipeline. See learn/04a-fastapi.md.

Run: uv run fastapi dev src/telco_support_agent/api.py   → open http://127.0.0.1:8000/docs
"""

import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated, Literal

import openai
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from telco_support_agent.config import get_settings
from telco_support_agent.generation import Market, RAGPipeline, RAGResponse
from telco_support_agent.llm import ExtractionError

# The app is an entry point, so it configures logging (library modules only create loggers).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TelcoX Support Agent",
    version="0.1.0",
    description="Answers TelcoX customer questions from policy documents, with citations.",
)


# --- Request and response models: the API contract ------------------------------------------


class AskRequest(BaseModel):
    """Body of POST /ask."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=500)
    market: Market


class CitationOut(BaseModel):
    label: str
    title: str
    heading: str
    chunk_id: str


class AskResponse(BaseModel):
    """What clients see. A deliberate subset of RAGResponse: no costs, scores or raw checks."""

    status: Literal["answered", "cannot_answer"]
    answer: str
    citations: list[CitationOut]
    checks_passed: bool
    latency_s: float

    @classmethod
    def from_rag(cls, r: RAGResponse) -> "AskResponse":
        return cls(
            status=r.status,
            answer=r.answer,
            citations=[CitationOut(**c.model_dump()) for c in r.citations],
            checks_passed=r.checks.ok,
            latency_s=r.latency_s,
        )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks: int
    vector_store: str
    embedding_model: str
    llm_model: str


# --- Dependencies ---------------------------------------------------------------------------


class NotReadyError(RuntimeError):
    """The service can't serve yet, e.g. the index hasn't been built."""


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Build the pipeline once, on first use, and share it across requests.

    lru_cache doesn't cache exceptions: if the index is missing, the next request tries again.
    Tests replace this with a fake via app.dependency_overrides.
    """
    try:
        return RAGPipeline()
    except Exception as e:  # missing index, bad config…
        raise NotReadyError(f"pipeline not ready: {e}") from e


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


# --- Middleware: a request ID and one log line per request ----------------------------------


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s %s %.0fms id=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - start) * 1000,
        request_id,
    )
    return response


# --- Error handling: turn exceptions into clean HTTP errors ---------------------------------


def _error(status_code: int, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


@app.exception_handler(NotReadyError)
async def not_ready(request: Request, exc: NotReadyError) -> JSONResponse:
    logger.error("not ready: %s", exc)
    return _error(503, "Service not ready. Try again shortly.")


@app.exception_handler(openai.APITimeoutError)
async def llm_timeout(request: Request, exc: openai.APITimeoutError) -> JSONResponse:
    return _error(504, "The language model timed out. Please retry.")


@app.exception_handler(openai.RateLimitError)
async def llm_rate_limited(request: Request, exc: openai.RateLimitError) -> JSONResponse:
    return _error(503, "Too many requests right now. Please retry.", {"Retry-After": "10"})


@app.exception_handler(openai.APIError)
async def llm_error(request: Request, exc: openai.APIError) -> JSONResponse:
    logger.exception("LLM API error")
    return _error(502, "The language model failed. Please retry.")


@app.exception_handler(ExtractionError)
async def llm_bad_output(request: Request, exc: ExtractionError) -> JSONResponse:
    logger.error("LLM returned no structured answer: %s", exc)
    return _error(502, "The language model returned an invalid answer. Please retry.")


# --- Routes ---------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(pipeline: PipelineDep) -> HealthResponse:
    """Readiness: the index is loaded. Deliberately makes no LLM call (fast and free)."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        chunks=len(pipeline.retriever.index),
        vector_store=settings.vector_store,
        embedding_model=pipeline.retriever.embedder.model,
        llm_model=settings.llm_model,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: PipelineDep) -> AskResponse:
    response = pipeline.ask(request.question, request.market)
    return AskResponse.from_rag(response)
