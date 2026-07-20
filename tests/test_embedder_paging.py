from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from scrutator.db.models import IndexRequest
from scrutator.search.embedder import EmbeddingError, embed_sparse, embed_texts
from scrutator.search.indexer import index_documents


def _response(data: list[dict]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": [{"index": index, **item} for index, item in enumerate(data)]}
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "page_sizes"),
    [(1, [1]), (64, [64]), (65, [64, 1]), (74, [64, 10]), (128, [64, 64]), (256, [64, 64, 64, 64])],
)
async def test_dense_embedding_pages_preserve_order_and_provider_limit(count, page_sizes):
    calls: list[list[str]] = []

    async def post(_url, *, json):
        page = json["input"]
        calls.append(page)
        return _response([{"embedding": [float(value)] * 1024} for value in page])

    client = AsyncMock()
    client.post.side_effect = post
    texts = [str(index) for index in range(count)]

    with patch("scrutator.search.embedder.get_client", return_value=client):
        result = await embed_texts(texts)

    assert [len(page) for page in calls] == page_sizes
    assert all(len(page) <= 64 for page in calls)
    assert [vector[0] for vector in result] == [float(index) for index in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "page_sizes"),
    [(1, [1]), (64, [64]), (65, [64, 1]), (74, [64, 10]), (128, [64, 64]), (256, [64, 64, 64, 64])],
)
async def test_sparse_embedding_pages_preserve_order_and_provider_limit(count, page_sizes):
    calls: list[list[str]] = []

    async def post(_url, *, json):
        page = json["input"]
        calls.append(page)
        return _response([{"sparse_weights": {value: float(value)}} for value in page])

    client = AsyncMock()
    client.post.side_effect = post
    texts = [str(index) for index in range(count)]

    with patch("scrutator.search.embedder.get_client", return_value=client):
        result = await embed_sparse(texts)

    assert [len(page) for page in calls] == page_sizes
    assert all(len(page) <= 64 for page in calls)
    assert [next(iter(vector)) for vector in result] == texts


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_embedding_page_cardinality_mismatch_fails_before_later_pages(mode):
    client = AsyncMock()
    field = "embedding" if mode == "dense" else "sparse_weights"
    value = [0.1] * 1024 if mode == "dense" else {"1": 0.1}
    client.post.return_value = _response([{field: value}] * 63)
    function = embed_texts if mode == "dense" else embed_sparse

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError, match="cardinality"),
    ):
        await function([str(index) for index in range(65)])

    assert client.post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_embedding_page_rejects_non_finite_values(mode):
    field = "embedding" if mode == "dense" else "sparse_weights"
    value = [float("nan")] * 1024 if mode == "dense" else {"1": float("inf")}
    client = AsyncMock()
    client.post.return_value = _response([{field: value}])
    function = embed_texts if mode == "dense" else embed_sparse

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError, match="invalid"),
    ):
        await function(["one"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "indices",
    [[1, 0], [0, 0], [0, 2], [0, "1"], [0, None]],
)
async def test_embedding_page_rejects_reordered_or_invalid_provider_indices(indices):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [{"index": index, "embedding": [float(position)] * 1024} for position, index in enumerate(indices)]
    }
    client = AsyncMock()
    client.post.return_value = response

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError, match="index order"),
    ):
        await embed_texts(["first", "second"])


@pytest.mark.asyncio
async def test_dense_embedding_page_rejects_float32_overflow():
    client = AsyncMock()
    client.post.return_value = _response([{"embedding": [1e308] * 1024}])

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError, match="invalid dense"),
    ):
        await embed_texts(["one"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_only_failed_second_page_is_retried(mode):
    calls: list[int] = []
    failed_once = False

    async def post(url, *, json):
        nonlocal failed_once
        page = json["input"]
        calls.append(len(page))
        if len(page) == 1 and not failed_once:
            failed_once = True
            raise httpx.ConnectError("bounded transport failure")
        field = "sparse_weights" if url.endswith("/sparse") else "embedding"
        values = [{field: {value: 0.1} if mode == "sparse" else [float(value)] * 1024} for value in page]
        return _response(values)

    client = AsyncMock()
    client.post.side_effect = post
    function = embed_sparse if mode == "sparse" else embed_texts

    with patch("scrutator.search.embedder.get_client", return_value=client):
        result = await function([str(index) for index in range(65)])

    assert len(result) == 65
    assert calls == [64, 1, 1]


@pytest.mark.asyncio
async def test_remote_protocol_error_is_retried_without_logging_details(caplog):
    client = AsyncMock()
    client.post.side_effect = [
        httpx.RemoteProtocolError("sensitive-transport-marker"),
        _response([{"embedding": [0.1] * 1024}]),
    ]

    with patch("scrutator.search.embedder.get_client", return_value=client):
        result = await embed_texts(["one"])

    assert result == [[0.1] * 1024]
    assert client.post.await_count == 2
    assert "error_type=RemoteProtocolError" in caplog.text
    assert "sensitive-transport-marker" not in caplog.text


@pytest.mark.asyncio
async def test_non_2xx_response_body_is_not_exposed():
    response = MagicMock()
    response.status_code = 400
    response.text = "sensitive-provider-marker"
    client = AsyncMock()
    client.post.return_value = response

    with (
        patch("scrutator.search.embedder.get_client", return_value=client),
        pytest.raises(EmbeddingError) as exc_info,
    ):
        await embed_texts(["one"])

    assert "sensitive-provider-marker" not in str(exc_info.value)
    assert exc_info.value.status_code == 400
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_later_dense_page_failure_persists_no_source():
    metadata = SimpleNamespace(
        source_type="markdown",
        heading_hierarchy=[],
        frontmatter={},
        wikilinks=[],
        tags=[],
        language="en",
        section=None,
    )
    chunks = [
        SimpleNamespace(
            id=f"00000000-0000-4000-8000-{index + 1:012d}",
            metadata=metadata,
            chunk_index=index,
            parent_id=None,
            content=str(index),
            content_hash=f"hash-{index}",
            token_count=1,
        )
        for index in range(65)
    ]
    first_page = [[0.1] * 1024 for _ in range(64)]

    with (
        patch("scrutator.search.indexer.chunk_document", return_value=SimpleNamespace(chunks=chunks)),
        patch(
            "scrutator.search.embedder._embed_dense_page",
            new=AsyncMock(side_effect=[first_page, EmbeddingError("page two failed")]),
        ),
        patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as namespace,
        patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as replace,
    ):
        results = await index_documents(
            [IndexRequest(content="source", source_path="large.md", namespace="self-improvement")]
        )

    assert results[0].status == "failed"
    assert results[0].error_code == "dense_embedding_failed"
    namespace.assert_not_awaited()
    replace.assert_not_awaited()
