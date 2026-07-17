from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from scrutator.config import Settings, settings
from scrutator.db.models import IndexRequest
from scrutator.search.embedder import EmbeddingError, embed_dense_sparse
from scrutator.search.indexer import index_document, index_documents


def _response(inputs: list[str]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {
                "index": index,
                "embedding": [float(value)] * 1024,
                "sparse_weights": {value: float(value)},
            }
            for index, value in enumerate(inputs)
        ]
    }
    return response


def _chunks(count: int) -> list[SimpleNamespace]:
    metadata = SimpleNamespace(
        source_type="markdown",
        heading_hierarchy=[],
        frontmatter={},
        wikilinks=[],
        tags=[],
        language="en",
        section=None,
    )
    return [
        SimpleNamespace(
            metadata=metadata,
            chunk_index=index,
            parent_id=None,
            content=str(index),
            content_hash=f"hash-{index}",
            token_count=1,
        )
        for index in range(count)
    ]


def test_dense_sparse_transport_defaults_off(monkeypatch):
    monkeypatch.delenv("SCRUTATOR_EMBEDDING_DENSE_SPARSE_ENABLED", raising=False)
    assert Settings().embedding_dense_sparse_enabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "page_sizes"),
    [(0, []), (64, [64]), (65, [64, 1]), (256, [64, 64, 64, 64])],
)
async def test_dense_sparse_pages_preserve_exact_order_and_provider_limit(count, page_sizes):
    calls: list[tuple[str, list[str]]] = []

    async def post(url, *, json):
        page = json["input"]
        calls.append((url, page))
        return _response(page)

    client = AsyncMock()
    client.post.side_effect = post
    texts = [str(index) for index in range(count)]

    with patch("scrutator.search.embedder.get_client", return_value=client):
        dense, sparse = await embed_dense_sparse(texts)

    assert [len(page) for _, page in calls] == page_sizes
    assert all(url.endswith("/v1/embeddings/dense-sparse") for url, _ in calls)
    assert all(len(page) <= 64 for _, page in calls)
    assert [vector[0] for vector in dense] == [float(index) for index in range(count)]
    assert [next(iter(vector)) for vector in sparse] == texts


@pytest.mark.asyncio
async def test_dense_sparse_retries_only_failed_page():
    calls: list[int] = []
    failed_once = False

    async def post(_url, *, json):
        nonlocal failed_once
        page = json["input"]
        calls.append(len(page))
        if len(page) == 1 and not failed_once:
            failed_once = True
            raise httpx.ConnectError("bounded transport failure")
        return _response(page)

    client = AsyncMock()
    client.post.side_effect = post
    with patch("scrutator.search.embedder.get_client", return_value=client):
        dense, sparse = await embed_dense_sparse([str(index) for index in range(65)])

    assert len(dense) == len(sparse) == 65
    assert calls == [64, 1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"data": []}, "cardinality"),
        (
            {
                "data": [
                    {"index": 1, "embedding": [0.1] * 1024, "sparse_weights": {"1": 0.1}},
                    {"index": 0, "embedding": [0.2] * 1024, "sparse_weights": {"2": 0.2}},
                ]
            },
            "index order",
        ),
        (
            {"data": [{"index": 0, "embedding": [1e308] * 1024, "sparse_weights": {"1": 0.1}}]},
            "dense",
        ),
        (
            {"data": [{"index": 0, "embedding": [3.40282348e38] * 1024, "sparse_weights": {"1": 0.1}}]},
            "dense",
        ),
        (
            {"data": [{"index": 0, "embedding": [0.1] * 1024, "sparse_weights": {"1": float("inf")}}]},
            "sparse",
        ),
    ],
)
async def test_dense_sparse_rejects_malformed_response(payload, match):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    client = AsyncMock()
    client.post.return_value = response
    texts = ["first", "second"] if len(payload.get("data", [])) == 2 else ["first"]

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError, match=match),
    ):
        await embed_dense_sparse(texts)


@pytest.mark.asyncio
async def test_dense_sparse_non_2xx_body_is_sanitized():
    response = MagicMock()
    response.status_code = 400
    response.text = "sensitive-provider-marker"
    client = AsyncMock()
    client.post.return_value = response

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError) as exc_info,
    ):
        await embed_dense_sparse(["secret input"])

    assert "sensitive-provider-marker" not in str(exc_info.value)
    assert "secret input" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_enabled_middle_page_failure_persists_nothing():
    first_page = ([[0.1] * 1024 for _ in range(64)], [{"1": 0.1} for _ in range(64)])
    original = settings.embedding_dense_sparse_enabled
    settings.embedding_dense_sparse_enabled = True
    try:
        with (
            patch("scrutator.search.indexer.chunk_document", return_value=SimpleNamespace(chunks=_chunks(256))),
            patch(
                "scrutator.search.embedder._embed_dense_sparse_page",
                new=AsyncMock(side_effect=[first_page, EmbeddingError("page two failed")]),
            ) as pages,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as namespace,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as replace,
        ):
            results = await index_documents(
                [IndexRequest(content="source", source_path="large.md", namespace="self-improvement")]
            )
    finally:
        settings.embedding_dense_sparse_enabled = original

    assert pages.await_count == 2
    assert results[0].status == "failed"
    namespace.assert_not_awaited()
    replace.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_on_routes_single_document_through_dense_sparse_transport():
    original = settings.embedding_dense_sparse_enabled
    settings.embedding_dense_sparse_enabled = True
    try:
        with (
            patch(
                "scrutator.search.indexer.embed_dense_sparse",
                new_callable=AsyncMock,
                return_value=([[0.1] * 1024], [{"1": 0.1}]),
            ) as combined,
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as dense,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch(
                "scrutator.search.indexer.replace_source_chunks_atomic",
                new_callable=AsyncMock,
                return_value=1,
            ) as replace,
        ):
            response = await index_document("source", "one.md", namespace="self-improvement")
    finally:
        settings.embedding_dense_sparse_enabled = original

    combined.assert_awaited_once_with(["source"])
    dense.assert_not_awaited()
    sparse.assert_not_awaited()
    replace.assert_awaited_once()
    assert response.chunks_indexed == 1


@pytest.mark.asyncio
async def test_flag_off_preserves_separate_dense_then_sparse_calls():
    dense_vectors = [[0.1] * 1024]
    sparse_vectors = [{"1": 0.1}]
    original = settings.embedding_dense_sparse_enabled
    settings.embedding_dense_sparse_enabled = False
    try:
        with (
            patch("scrutator.search.indexer.embed_dense_sparse", new_callable=AsyncMock) as combined,
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=dense_vectors) as dense,
            patch(
                "scrutator.search.indexer.embed_sparse", new_callable=AsyncMock, return_value=sparse_vectors
            ) as sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch(
                "scrutator.search.indexer.replace_source_chunks_atomic",
                new_callable=AsyncMock,
                return_value=1,
            ) as replace,
        ):
            results = await index_documents(
                [IndexRequest(content="source", source_path="one.md", namespace="self-improvement")]
            )
    finally:
        settings.embedding_dense_sparse_enabled = original

    combined.assert_not_awaited()
    dense.assert_awaited_once_with(["source"])
    sparse.assert_awaited_once_with(["source"])
    replace.assert_awaited_once()
    assert results[0].status == "succeeded"
