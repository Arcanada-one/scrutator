"""SRCH-0026 bounded batch indexing contract."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.db.models import BatchIndexSucceeded
from scrutator.health import app
from scrutator.search import indexer as indexer_module


def test_giant_markdown_serialization_preserves_parent_ids():
    result = indexer_module.chunk_document(
        "# Giant\n\n" + ("word " * 3000),
        "giant.md",
        max_tokens=64,
        overlap_tokens=8,
    )
    serialized = indexer_module._chunk_dicts(result, "test", "giant.md")

    emitted = set()
    assert len(serialized) > 1
    for source, stored in zip(result.chunks, serialized, strict=True):
        assert stored["id"] == source.id
        if stored["parent_id"] is not None:
            assert stored["parent_id"] in emitted
        emitted.add(stored["id"])


def test_batch_endpoint_preserves_single_document_response_order():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    try:
        with (
            patch("scrutator.health.index_documents", new_callable=AsyncMock) as index,
            TestClient(app) as client,
        ):
            index.return_value = [
                BatchIndexSucceeded(
                    chunks_indexed=1,
                    source_path="reflection-LTM-0001.md",
                )
            ]
            response = client.post(
                "/v1/index/batch",
                json={
                    "documents": [
                        {
                            "content": "# Reflection\n\nA durable lesson.",
                            "source_path": "reflection-LTM-0001.md",
                            "namespace": "self-improvement",
                        }
                    ]
                },
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "source_path": "reflection-LTM-0001.md",
                "status": "succeeded",
                "chunks_indexed": 1,
            }
        ]
    }


def test_batch_rejects_duplicate_source_paths_before_indexing():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    body = {
        "documents": [
            {"content": "first", "source_path": "same.md", "namespace": "self-improvement"},
            {"content": "second", "source_path": "same.md", "namespace": "self-improvement"},
        ]
    }
    try:
        with (
            patch("scrutator.health.index_documents", new_callable=AsyncMock) as index,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json=body,
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 422
    index.assert_not_awaited()


def test_batch_rejects_mixed_namespaces_before_indexing():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement,wiki"
    body = {
        "documents": [
            {"content": "first", "source_path": "a.md", "namespace": "self-improvement"},
            {"content": "second", "source_path": "b.md", "namespace": "wiki"},
        ]
    }
    try:
        with (
            patch("scrutator.health.index_documents", new_callable=AsyncMock) as index,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json=body,
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 422
    index.assert_not_awaited()


def test_batch_rejects_document_over_byte_cap():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    try:
        with patch("scrutator.health.index_documents", new_callable=AsyncMock), TestClient(app) as client:
            response = client.post(
                "/v1/index/batch",
                json={
                    "documents": [
                        {
                            "content": "x" * 262_145,
                            "source_path": "too-large.md",
                            "namespace": "self-improvement",
                        }
                    ]
                },
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 422


def test_batch_rejects_actual_request_body_over_byte_cap():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    documents = [
        {
            "content": "x" * 262_100,
            "source_path": f"{index}.md",
            "namespace": "self-improvement",
        }
        for index in range(4)
    ]
    try:
        with patch("scrutator.health.index_documents", new_callable=AsyncMock), TestClient(app) as client:
            response = client.post(
                "/v1/index/batch",
                json={"documents": documents},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 413


def test_batch_packs_chunks_into_one_dense_and_one_sparse_call_in_document_order():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    documents = [
        {"content": "first lesson", "source_path": "a.md", "namespace": "self-improvement"},
        {"content": "second lesson", "source_path": "b.md", "namespace": "self-improvement"},
    ]
    dense_vectors = [[0.1] * 1024, [0.2] * 1024]
    sparse_vectors = [{"1": 0.1}, {"2": 0.2}]
    try:
        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=dense_vectors) as dense,
            patch(
                "scrutator.search.indexer.embed_sparse",
                new_callable=AsyncMock,
                return_value=sparse_vectors,
            ) as sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch(
                "scrutator.search.indexer.replace_source_chunks_atomic",
                new_callable=AsyncMock,
                return_value=1,
            ) as replace,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": documents},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 200
    assert [result["source_path"] for result in response.json()["results"]] == ["a.md", "b.md"]
    dense.assert_awaited_once_with(["first lesson", "second lesson"])
    sparse.assert_awaited_once_with(["first lesson", "second lesson"])
    assert replace.await_count == 2
    assert replace.await_args_list[0].args[1] == [[0.1] * 1024]
    assert replace.await_args_list[0].args[2] == [{"1": 0.1}]
    assert replace.await_args_list[1].args[1] == [[0.2] * 1024]
    assert replace.await_args_list[1].args[2] == [{"2": 0.2}]


@pytest.mark.parametrize(
    ("dense_vectors", "expected_code"),
    [
        ([[0.1] * 1024], "invalid_dense_embeddings"),
        ([[float("nan")] * 1024, [0.2] * 1024], "invalid_dense_embeddings"),
        ([[0.1] * 12, [0.2] * 1024], "invalid_dense_embeddings"),
    ],
)
def test_batch_rejects_malformed_dense_vectors_without_persisting(dense_vectors, expected_code):
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    documents = [
        {"content": "first", "source_path": "a.md", "namespace": "self-improvement"},
        {"content": "second", "source_path": "b.md", "namespace": "self-improvement"},
    ]
    try:
        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=dense_vectors),
            patch(
                "scrutator.search.indexer.embed_sparse",
                new_callable=AsyncMock,
                return_value=[{"1": 0.1}, {"2": 0.2}],
            ),
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as replace,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": documents},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 200
    assert [result["error_code"] for result in response.json()["results"]] == [expected_code, expected_code]
    replace.assert_not_awaited()


def test_batch_rejects_malformed_sparse_vectors_without_persisting():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    documents = [
        {"content": "first", "source_path": "a.md", "namespace": "self-improvement"},
        {"content": "second", "source_path": "b.md", "namespace": "self-improvement"},
    ]
    try:
        with (
            patch(
                "scrutator.search.indexer.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024, [0.2] * 1024],
            ),
            patch(
                "scrutator.search.indexer.embed_sparse",
                new_callable=AsyncMock,
                return_value=[{"1": float("inf")}, {"2": 0.2}],
            ),
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as replace,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": documents},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert [result["error_code"] for result in response.json()["results"]] == [
        "invalid_sparse_embeddings",
        "invalid_sparse_embeddings",
    ]
    replace.assert_not_awaited()


@pytest.mark.parametrize(
    ("failing_stage", "expected_code"),
    [("dense", "dense_embedding_failed"), ("sparse", "sparse_embedding_failed")],
)
def test_batch_converts_embedding_failures_to_bounded_per_source_codes(failing_stage, expected_code, caplog):
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    dense = AsyncMock(return_value=[[0.1] * 1024])
    sparse = AsyncMock(return_value=[{"1": 0.1}])
    if failing_stage == "dense":
        dense.side_effect = RuntimeError("sensitive-embedding-marker")
    else:
        sparse.side_effect = RuntimeError("sensitive-embedding-marker")
    try:
        with (
            patch("scrutator.search.indexer.embed_texts", new=dense),
            patch("scrutator.search.indexer.embed_sparse", new=sparse),
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as replace,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": [{"content": "first", "source_path": "a.md", "namespace": "self-improvement"}]},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.json()["results"] == [{"source_path": "a.md", "status": "failed", "error_code": expected_code}]
    assert "sensitive-embedding-marker" not in caplog.text
    replace.assert_not_awaited()


def test_batch_returns_ordered_partial_results_and_sanitizes_persistence_error(caplog):
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    documents = [
        {"content": "first", "source_path": "a.md", "namespace": "self-improvement"},
        {"content": "second", "source_path": "b.md", "namespace": "self-improvement"},
    ]
    try:
        with (
            patch(
                "scrutator.search.indexer.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024, [0.2] * 1024],
            ),
            patch(
                "scrutator.search.indexer.embed_sparse",
                new_callable=AsyncMock,
                return_value=[{"1": 0.1}, {"2": 0.2}],
            ),
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch(
                "scrutator.search.indexer.replace_source_chunks_atomic",
                new_callable=AsyncMock,
                side_effect=[1, RuntimeError("sensitive-content-marker")],
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": documents},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.json()["results"] == [
        {"source_path": "a.md", "status": "succeeded", "chunks_indexed": 1},
        {"source_path": "b.md", "status": "failed", "error_code": "persistence_failed"},
    ]
    assert "sensitive-content-marker" not in caplog.text


def test_batch_sanitizes_common_namespace_persistence_failure(caplog):
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    try:
        with (
            patch(
                "scrutator.search.indexer.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ),
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock, return_value=[{"1": 0.1}]),
            patch(
                "scrutator.search.indexer.upsert_namespace",
                new_callable=AsyncMock,
                side_effect=RuntimeError("sensitive-namespace-marker"),
            ),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": [{"content": "first", "source_path": "a.md", "namespace": "self-improvement"}]},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"source_path": "a.md", "status": "failed", "error_code": "persistence_failed"}
    ]
    assert "sensitive-namespace-marker" not in caplog.text


def test_batch_rejects_total_chunk_cap_before_embedding():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    documents = [
        {"content": "first", "source_path": "a.md", "namespace": "self-improvement"},
        {"content": "second", "source_path": "b.md", "namespace": "self-improvement"},
    ]
    try:
        with (
            patch("scrutator.search.indexer.INDEX_BATCH_MAX_CHUNKS", 1, create=True),
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as dense,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={"documents": documents},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 422
    dense.assert_not_awaited()


def test_batch_rejects_total_token_cap_before_embedding():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    try:
        with (
            patch("scrutator.search.indexer.INDEX_BATCH_MAX_TOKENS", 1),
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as dense,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={
                    "documents": [
                        {
                            "content": "two tokens",
                            "source_path": "a.md",
                            "namespace": "self-improvement",
                        }
                    ]
                },
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 422
    dense.assert_not_awaited()


def test_batch_chunk_failure_isolated_to_one_source():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    real_chunk_document = indexer_module.chunk_document

    def chunk_or_fail(**kwargs):
        if kwargs["source_path"] == "bad.md":
            raise RuntimeError("sensitive-chunk-marker")
        return real_chunk_document(**kwargs)

    try:
        with (
            patch("scrutator.search.indexer.chunk_document", side_effect=chunk_or_fail),
            patch(
                "scrutator.search.indexer.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1] * 1024],
            ) as dense,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock, return_value=[{"1": 0.1}]),
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock, return_value=1),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                json={
                    "documents": [
                        {"content": "bad", "source_path": "bad.md", "namespace": "self-improvement"},
                        {"content": "good", "source_path": "good.md", "namespace": "self-improvement"},
                    ]
                },
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"source_path": "bad.md", "status": "failed", "error_code": "chunking_failed"},
        {"source_path": "good.md", "status": "succeeded", "chunks_indexed": 1},
    ]
    dense.assert_awaited_once_with(["good"])


def test_batch_requires_feeder_capability_and_enforces_its_namespace_scope():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    body = {"documents": [{"content": "x", "source_path": "x.md", "namespace": "wiki"}]}
    try:
        with TestClient(app) as client:
            missing = client.post("/v1/index/batch", json=body)
            outside = client.post(
                "/v1/index/batch",
                json=body,
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert missing.status_code == 401
    assert outside.status_code == 403
