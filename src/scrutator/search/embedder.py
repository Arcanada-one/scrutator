"""Embedding client — async httpx calls to Embedding API (arcana-db:8300).

Uses a singleton httpx.AsyncClient to avoid connection pool exhaustion.
SRCH-0020: per-call client creation caused 503 after 2-3 index requests.
"""

from __future__ import annotations

import asyncio
import logging
import math

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scrutator.config import settings

logger = logging.getLogger(__name__)

# The shared Embedding API rejects HTTP batches larger than 64. Scrutator's
# higher-level index pack may contain up to 256 chunks, so transport requests
# use bounded page concurrency without weakening the indexer's independent caps.
_EMBEDDING_API_MAX_BATCH_SIZE = 64
_DENSE_SPARSE_PAGE_CONCURRENCY = 2
_COLBERT_API_MAX_BATCH_SIZE = 16
_DENSE_DIMENSIONS = 1024
_FLOAT32_MAX = 3.4028235e38
_EXACT_FLOAT32_MAX = 3.4028234663852886e38

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

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ── Retry decorator ─────────────────────────────────────────────────

_RETRYABLE = (httpx.TransportError,)


def _log_retry(retry_state, max_attempts: int) -> None:
    exception = retry_state.outcome.exception()
    logger.warning(
        "Embedding retry %d/%d: error_type=%s status_code=none",
        retry_state.attempt_number,
        max_attempts,
        type(exception).__name__,
    )


def _with_retry_attempts(fn, max_attempts: int):
    """Apply a bounded tenacity policy to an async embedding function."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=lambda retry_state: _log_retry(retry_state, max_attempts),
    )(fn)


def _with_retry(fn):
    return _with_retry_attempts(fn, settings.embedding_max_retries)


def _with_dense_sparse_retry(fn):
    return _with_retry_attempts(fn, settings.embedding_dense_sparse_max_retries)


# ── Public API ──────────────────────────────────────────────────────


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _finite_dense_number(value: object) -> bool:
    return _finite_number(value) and abs(float(value)) <= _FLOAT32_MAX


def _finite_dense_sparse_number(value: object) -> bool:
    return _finite_number(value) and abs(float(value)) <= _EXACT_FLOAT32_MAX


def _response_data(response: httpx.Response, expected_count: int) -> list[dict]:
    try:
        data = response.json()["data"]
    except Exception as exc:
        raise EmbeddingError("Embedding API returned an invalid response") from exc
    if not isinstance(data, list) or len(data) != expected_count:
        raise EmbeddingError("Embedding API response cardinality mismatch")
    if not all(isinstance(item, dict) for item in data):
        raise EmbeddingError("Embedding API returned an invalid response")
    indices = [item.get("index") for item in data]
    if not all(type(index) is int for index in indices) or indices != list(range(expected_count)):
        raise EmbeddingError("Embedding API response index order mismatch")
    return data


@_with_retry
async def _embed_dense_page(texts: list[str]) -> list[list[float]]:
    client = await get_client()
    response = await client.post(
        f"{settings.embedding_api_url}/v1/embeddings",
        json={"input": texts},
    )
    if response.status_code != 200:
        raise EmbeddingError(
            f"Embedding API returned status {response.status_code}",
            status_code=response.status_code,
        )

    data = _response_data(response, len(texts))
    embeddings = [item.get("embedding") for item in data]
    if not all(
        isinstance(vector, list)
        and len(vector) == _DENSE_DIMENSIONS
        and all(_finite_dense_number(value) for value in vector)
        for vector in embeddings
    ):
        raise EmbeddingError("Embedding API returned invalid dense embeddings")
    return embeddings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get ordered dense embeddings, paging to the provider's 64-item cap."""
    embeddings: list[list[float]] = []
    for offset in range(0, len(texts), _EMBEDDING_API_MAX_BATCH_SIZE):
        embeddings.extend(await _embed_dense_page(texts[offset : offset + _EMBEDDING_API_MAX_BATCH_SIZE]))
    return embeddings


@_with_retry
async def _embed_sparse_page(texts: list[str]) -> list[dict[str, float]]:
    client = await get_client()
    response = await client.post(
        f"{settings.embedding_api_url}/v1/embeddings/sparse",
        json={"input": texts},
    )
    if response.status_code != 200:
        raise EmbeddingError(
            f"Sparse Embedding API returned status {response.status_code}",
            status_code=response.status_code,
        )

    data = _response_data(response, len(texts))
    embeddings = [item.get("sparse_weights") for item in data]
    if not all(
        isinstance(vector, dict)
        and all(isinstance(token, str) and _finite_number(weight) for token, weight in vector.items())
        for vector in embeddings
    ):
        raise EmbeddingError("Embedding API returned invalid sparse embeddings")
    return embeddings


async def embed_sparse(texts: list[str]) -> list[dict[str, float]]:
    """Get ordered sparse embeddings, paging to the provider's 64-item cap."""
    embeddings: list[dict[str, float]] = []
    for offset in range(0, len(texts), _EMBEDDING_API_MAX_BATCH_SIZE):
        embeddings.extend(await _embed_sparse_page(texts[offset : offset + _EMBEDDING_API_MAX_BATCH_SIZE]))
    return embeddings


@_with_dense_sparse_retry
async def _embed_dense_sparse_page(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    client = await get_client()
    response = await client.post(
        f"{settings.embedding_api_url}/v1/embeddings/dense-sparse",
        json={"input": texts},
        timeout=settings.embedding_dense_sparse_timeout,
    )
    if response.status_code != 200:
        raise EmbeddingError(f"Dense-sparse Embedding API returned status {response.status_code}")

    data = _response_data(response, len(texts))
    dense = [item.get("dense") for item in data]
    if not all(
        isinstance(vector, list)
        and len(vector) == _DENSE_DIMENSIONS
        and all(_finite_dense_sparse_number(value) for value in vector)
        for vector in dense
    ):
        raise EmbeddingError("Embedding API returned invalid dense embeddings")

    sparse = [item.get("sparse_weights") for item in data]
    if not all(
        isinstance(vector, dict)
        and all(isinstance(token, str) and _finite_number(weight) for token, weight in vector.items())
        for vector in sparse
    ):
        raise EmbeddingError("Embedding API returned invalid sparse embeddings")
    return dense, sparse


async def embed_dense_sparse(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    """Get ordered paired vectors with bounded page-level parallelism."""
    pages = [
        texts[offset : offset + _EMBEDDING_API_MAX_BATCH_SIZE]
        for offset in range(0, len(texts), _EMBEDDING_API_MAX_BATCH_SIZE)
    ]
    semaphore = asyncio.Semaphore(_DENSE_SPARSE_PAGE_CONCURRENCY)

    async def embed_page(page: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        async with semaphore:
            return await _embed_dense_sparse_page(page)

    page_results = await asyncio.gather(*(embed_page(page) for page in pages))
    dense: list[list[float]] = []
    sparse: list[dict[str, float]] = []
    for page_dense, page_sparse in page_results:
        dense.extend(page_dense)
        sparse.extend(page_sparse)
    return dense, sparse


async def embed_single(text: str) -> list[float]:
    """Get dense embedding for a single text."""
    results = await embed_texts([text])
    return results[0]


@_with_retry
async def _embed_colbert_page(texts: list[str]) -> list[list[list[float]]]:
    client = await get_client()
    response = await client.post(
        f"{settings.embedding_api_url}/v1/embeddings/colbert",
        json={"input": texts},
    )
    if response.status_code != 200:
        raise EmbeddingError(
            f"ColBERT Embedding API returned status {response.status_code}",
            status_code=response.status_code,
        )

    data = _response_data(response, len(texts))
    return [item["colbert_vecs"] for item in data]


async def embed_colbert(texts: list[str]) -> list[list[list[float]]]:
    """Token-level ColBERT multi-vectors from the Embedding API.

    Returns list (per text) of list (per token) of 1024-dim vectors.
    Pages requests to the provider's independent 16-item ColBERT cap.
    Field: data[i].colbert_vecs (probe-confirmed 2026-06-22).
    """
    embeddings: list[list[list[float]]] = []
    for offset in range(0, len(texts), _COLBERT_API_MAX_BATCH_SIZE):
        embeddings.extend(await _embed_colbert_page(texts[offset : offset + _COLBERT_API_MAX_BATCH_SIZE]))
    return embeddings
