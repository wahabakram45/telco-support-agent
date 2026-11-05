from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from telco_support_agent.api import NotReadyError, app, get_pipeline
from telco_support_agent.generation import Checks, Citation, RAGResponse
from telco_support_agent.llm import ExtractionError


class FakeIndex:
    def __len__(self) -> int:
        return 201


class FakePipeline:
    """Stands in for RAGPipeline: no index, no API calls. Raises `error` if one is set."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.retriever = SimpleNamespace(
            index=FakeIndex(), embedder=SimpleNamespace(model="emb-model")
        )

    def ask(self, question: str, market: str) -> RAGResponse:
        self.calls.append((question, market))
        if self.error:
            raise self.error
        return RAGResponse(
            question=question,
            market=market,
            status="answered",
            answer="The GCC Day Pass costs 25 SAR.",
            citations=[
                Citation(label="S1", chunk_id="faq-roaming#2", title="Roaming FAQs", heading="GCC")
            ],
            retrieved=[("faq-roaming#2", 0.71)],
            checks=Checks(),
            cost_usd=0.0003,
            latency_s=1.2,
        )


@pytest.fixture
def fake():
    pipeline = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield pipeline
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


GOOD = {"question": "How much is the GCC day pass?", "market": "KSA"}
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


# --- provided: health, middleware, docs ---


def test_health(fake, client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["chunks"] == 201 and response.json()["embedding_model"] == "emb-model"


def test_health_not_ready_is_503(client):
    def not_ready():
        raise NotReadyError("no index")

    app.dependency_overrides[get_pipeline] = not_ready
    try:
        response = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "no index" not in response.text  # internals stay in the logs, not in the response


def test_request_id_is_echoed_or_generated(fake, client):
    assert client.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"
    assert len(client.get("/health").headers["X-Request-ID"]) == 12


# --- your Task 2: POST /ask ---


def test_ask_returns_the_public_answer(fake, client):
    response = client.post("/ask", json=GOOD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered" and "25 SAR" in body["answer"]
    assert body["citations"][0]["chunk_id"] == "faq-roaming#2"
    assert body["checks_passed"] is True
    assert "cost_usd" not in body and "retrieved" not in body  # internal fields stay internal
    assert fake.calls == [(GOOD["question"], "KSA")]


def test_ask_is_documented_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/ask"]
    assert "AskResponse" in str(paths["/ask"]["post"]["responses"]["200"])


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (openai.APITimeoutError(request=REQUEST), 504),
        (
            openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=REQUEST), body=None
            ),
            503,
        ),
        (openai.APIConnectionError(request=REQUEST), 502),
        (ExtractionError("refused"), 502),
    ],
)
def test_llm_failures_become_clean_errors(client, error, status):
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline(error)
    try:
        response = client.post("/ask", json=GOOD)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == status
    assert set(response.json()) == {"detail"}


# --- your Task 1: AskRequest validation ---


def test_question_whitespace_is_stripped(fake, client):
    client.post("/ask", json={"question": "  How much?  ", "market": "UAE"})
    assert fake.calls == [("How much?", "UAE")]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "How much?"},  # market missing
        {"question": "How much?", "market": "PK"},  # unknown market
        {"question": "", "market": "KSA"},  # empty
        {"question": "    ", "market": "KSA"},  # only spaces
        {"question": "x" * 501, "market": "KSA"},  # too long
        {"question": "How much?", "market": "KSA", "customer_id": "42"},  # unexpected field
    ],
)
def test_invalid_requests_are_422_and_never_reach_the_pipeline(fake, client, body):
    response = client.post("/ask", json=body)
    assert response.status_code == 422
    assert fake.calls == []


# --- live: the real pipeline (costs a fraction of a cent; run with -m live) ---


@pytest.mark.live
def test_live_ask(client):
    response = client.post("/ask", json=GOOD)
    assert response.status_code == 200 and "25" in response.json()["answer"]
