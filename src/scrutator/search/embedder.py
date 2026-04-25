"""Embedding client — async httpx calls to Embedding API (arcana-db:8300).

Uses a singleton httpx.AsyncClient to avoid connection pool exhaustion.
SRCH-0020: per-call client creation caused 503 after 2-3 index requests.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scrutator.config import settings

logger = logging.getLogger(__name__)

# ── Singleton httpx client ──────────────────────────────────────────

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the singleton httpx client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.embedding_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )
    return _client


async def close_client() -> None:
    """Close the singleton httpx client (call on shutdown)."""
    global _client  # noqa: PLW0603
    if _client is not None:
        await _client.aclose()
        _client = None


# ── Exceptions ──────────────────────────────────────────────────────


class EmbeddingError(Exception):
    """Raised when the Embedding API returns an error."""


# ── Retry decorator ─────────────────────────────────────────────────

_RETRYABLE = (httpx.ConnectError, httpx.TimeoutException, httpx.PoolTimeout)


def _log_retry(retry_state) -> None:
    logger.warning(
        "Embedding retry %d/%d: %s",
        retry_state.attempt_number,
        settings.embedding_max_retries,
        retry_state.outcome.exception(),
    )


def _with_retry(fn):
    """Apply tenacity retry to an async embedding function."""
    return retry(
        stop=stop_after_attempt(settings.embedding_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=_log_retry,
    )(fn)


# ── Public API ──────────────────────────────────────────────────────


@_with_retry
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get dense embeddings for a list of texts from the Embedding API.

    Returns list of 1024-dim vectors.
    """
    if not texts:
        return []

    client = await get_client()
    response = await client.post(
        f"{settings.embedding_api_url}/v1/embeddings",
        json={"input": texts},
    )
    if response.status_code != 200:
        raise EmbeddingError(f"Embedding API returned {response.status_code}: {response.text}")

    data = response.json()
    return [item["embedding"] for item in data["data"]]


@_with_retry
async def embed_sparse(texts: list[str]) -> list[dict[str, float]]:
    """Get sparse (lexical weight) embeddings from the Embedding API.

    Returns list of {token_id: weight} dicts.
    """
    if not texts:
        return []

    client = await get_client()
    response = await client.post(
        f"{settings.embedding_api_url}/v1/embeddings/sparse",
        json={"input": texts},
    )
    if response.status_code != 200:
        raise EmbeddingError(f"Sparse Embedding API returned {response.status_code}: {response.text}")

    data = response.json()
    return [item["sparse_weights"] for item in data["data"]]


async def embed_single(text: str) -> list[float]:
    """Get dense embedding for a single text."""
    results = await embed_texts([text])
    return results[0]
