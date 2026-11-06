# syntax=docker/dockerfile:1
# Image for the TelcoX support API. See learn/04b-docker.md.
#   Build: docker build -t telco-support-agent .
#   Run:   docker run --rm -p 8000:8000 --env-file .env telco-support-agent

# --- Stage 1: builder — install dependencies into a virtualenv ----------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

# Compile .pyc files now (faster startup), copy files instead of hard-linking (works across
# layers), and use the image's Python instead of downloading one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# TODO(you) — lesson 04b, Task 1: install dependencies and our code, in a cache-friendly order.
#   1. copy only the files that list dependencies (pyproject.toml, uv.lock)
#   2. install dependencies only:        uv sync --locked --no-dev --no-install-project
#   3. copy our code (src/)
#   4. install our package, non-editable: uv sync --locked --no-dev --no-editable

# --- Stage 2: runtime — only what's needed to serve ---------------------------------------
FROM python:3.12-slim

# A non-root user, and the directories the app writes to (logs, embedding cache).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/logs /app/.cache \
    && chown -R app:app /app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
# The prebuilt index ships inside the image, so code and index are versioned together.
COPY --chown=app:app data/chroma ./data/chroma

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# TODO(you) — lesson 04b, Task 2: run as the `app` user, and start uvicorn serving
# telco_support_agent.api:app on port 8000, reachable from outside the container (exec form).
