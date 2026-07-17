from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.health import app
from tests.conftest import override_tenant_context

ANONYMOUS = TenantContext(
    principal_id="anonymous",
    principal_type="service",
    allowed_namespace_ids=frozenset(),
    allowed_namespace_names=frozenset(),
)
READER = TenantContext(
    principal_id="muneral-reader",
    principal_type="service",
    allowed_namespace_ids=frozenset({7}),
    allowed_namespace_names=frozenset({"muneral"}),
)
BODY = {"content": "safe content", "source_path": "muneral://task/1", "namespace": "muneral"}


def _post(ctx, *, token=None, body=None):
    headers = {"X-LTM-Writer-Token": token} if token is not None else {}
    with (
        override_tenant_context(app, ctx),
        patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock) as upsert,
        TestClient(app) as client,
    ):
        response = client.post("/v1/ltm/ingest", json=body or BODY, headers=headers)
    return response, upsert


def test_anonymous_ingest_requires_writer_token_before_namespace_upsert():
    response, upsert = _post(ANONYMOUS)
    assert response.status_code == 401
    upsert.assert_not_awaited()


def test_bearer_reader_grant_does_not_authorize_ingest():
    response, upsert = _post(READER)
    assert response.status_code == 401
    upsert.assert_not_awaited()


def test_bad_writer_token_is_rejected_before_namespace_upsert():
    original = settings.ltm_writer_token
    settings.ltm_writer_token = "correct-secret"
    try:
        response, upsert = _post(READER, token="wrong-secret")
    finally:
        settings.ltm_writer_token = original
    assert response.status_code == 401
    upsert.assert_not_awaited()


def test_non_ascii_writer_token_is_rejected_instead_of_crashing():
    original = settings.ltm_writer_token
    settings.ltm_writer_token = "correct-secret"
    try:
        response, upsert = _post(READER, token=b"\xff")
    finally:
        settings.ltm_writer_token = original
    assert response.status_code == 401
    upsert.assert_not_awaited()


def test_feeder_token_is_not_accepted_as_ltm_writer_token():
    original = (settings.ltm_writer_token, settings.feeder_token)
    settings.ltm_writer_token = "ltm-secret"
    settings.feeder_token = "feeder-secret"
    try:
        response, upsert = _post(READER, token="feeder-secret")
    finally:
        settings.ltm_writer_token, settings.feeder_token = original
    assert response.status_code == 401
    upsert.assert_not_awaited()


def test_out_of_scope_namespace_is_rejected_before_namespace_upsert():
    original = (settings.ltm_writer_token, settings.ltm_writer_namespaces)
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    try:
        response, upsert = _post(READER, token="ltm-secret", body={**BODY, "namespace": "other"})
    finally:
        settings.ltm_writer_token, settings.ltm_writer_namespaces = original
    assert response.status_code == 403
    upsert.assert_not_awaited()


def test_structured_ingest_uses_same_writer_authorization():
    structured_body = {
        **BODY,
        "structured_graph": {
            "schema_version": 1,
            "content_hash": "a" * 64,
            "entities": [{"name": "MUN:task-1", "entity_type": "task", "properties": {}}],
            "edges": [],
        },
    }
    response, upsert = _post(READER, body=structured_body)
    assert response.status_code == 401
    upsert.assert_not_awaited()


def test_valid_scoped_writer_token_enters_existing_generic_ingest_pipeline():
    original = (settings.ltm_writer_token, settings.ltm_writer_namespaces)
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    pool = MagicMock()
    try:
        with (
            override_tenant_context(app, READER),
            patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock, return_value=7) as upsert,
            patch("scrutator.ltm.router.repository.create_ltm_job", new_callable=AsyncMock, return_value="job-1"),
            patch("scrutator.ltm.router.repository.update_ltm_job", new_callable=AsyncMock),
            patch(
                "scrutator.ltm.router.repository.get_chunk_ids_by_source", new_callable=AsyncMock, return_value=[]
            ) as get_chunk_ids,
            patch(
                "scrutator.ltm.router.repository.get_entity_names_for_namespace",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "scrutator.search.indexer.index_document",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(chunks_indexed=0),
            ) as index_document,
            patch("scrutator.ltm.router._create_llm_client"),
            patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=pool),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/ltm/ingest",
                json=BODY,
                headers={"X-LTM-Writer-Token": "ltm-secret"},
            )
    finally:
        settings.ltm_writer_token, settings.ltm_writer_namespaces = original

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "status": "done",
        "entities_upserted": 0,
        "edges_upserted": 0,
        "idempotent_noop": False,
        "indexed": True,
        "total_chunks": 0,
        "enrichment": "complete",
        "enrichment_error": None,
    }
    upsert.assert_awaited_once_with("muneral")
    get_chunk_ids.assert_awaited_once_with("muneral://task/1", 7)
    index_document.assert_awaited_once()


def test_generic_ingest_returns_partial_when_enrichment_fails_after_indexing():
    original = (settings.ltm_writer_token, settings.ltm_writer_namespaces)
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    try:
        with (
            override_tenant_context(app, READER),
            patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.ltm.router.repository.create_ltm_job", new_callable=AsyncMock, return_value="job-1"),
            patch("scrutator.ltm.router.repository.update_ltm_job", new_callable=AsyncMock) as update_job,
            patch(
                "scrutator.ltm.router.repository.get_chunk_ids_by_source",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "scrutator.ltm.router.repository.get_entity_names_for_namespace",
                new_callable=AsyncMock,
                side_effect=RuntimeError("MC unavailable bearer-secret-sentinel"),
            ),
            patch(
                "scrutator.search.indexer.index_document",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(chunks_indexed=2),
            ),
            patch("scrutator.ltm.router._create_llm_client"),
            patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=MagicMock()),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/ltm/ingest",
                json=BODY,
                headers={"X-LTM-Writer-Token": "ltm-secret"},
            )
    finally:
        settings.ltm_writer_token, settings.ltm_writer_namespaces = original

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "status": "partial",
        "entities_upserted": 0,
        "edges_upserted": 0,
        "idempotent_noop": False,
        "indexed": True,
        "total_chunks": 2,
        "enrichment": "failed",
        "enrichment_error": "enrichment_failed",
    }
    assert update_job.await_args_list[-1].kwargs == {
        "status": "partial",
        "current_step": "enrichment_failed",
        "total_chunks": 2,
        "processed_chunks": 0,
        "error": "enrichment_failed",
    }
    assert "bearer-secret-sentinel" not in response.text


def test_generic_ingest_returns_500_when_critical_indexing_fails():
    original = (settings.ltm_writer_token, settings.ltm_writer_namespaces)
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    try:
        with (
            override_tenant_context(app, READER),
            patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.ltm.router.repository.create_ltm_job", new_callable=AsyncMock, return_value="job-1"),
            patch("scrutator.ltm.router.repository.update_ltm_job", new_callable=AsyncMock) as update_job,
            patch(
                "scrutator.search.indexer.index_document",
                new_callable=AsyncMock,
                side_effect=RuntimeError("embedding unavailable"),
            ),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/ltm/ingest",
                json=BODY,
                headers={"X-LTM-Writer-Token": "ltm-secret"},
            )
    finally:
        settings.ltm_writer_token, settings.ltm_writer_namespaces = original

    assert response.status_code == 500
    assert response.json() == {"detail": "Ingest failed: indexing error"}
    assert update_job.await_args_list[-1].kwargs == {
        "status": "failed",
        "error": "indexing_failed",
    }


def test_partial_job_preserves_processed_chunk_count():
    original = (settings.ltm_writer_token, settings.ltm_writer_namespaces)
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    pipeline = MagicMock()
    pipeline.process_chunk = AsyncMock(side_effect=[None, RuntimeError("provider unavailable")])
    conn = AsyncMock()
    conn.fetchrow.return_value = {"content": "chunk content"}
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    try:
        with (
            override_tenant_context(app, READER),
            patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.ltm.router.repository.create_ltm_job", new_callable=AsyncMock, return_value="job-1"),
            patch("scrutator.ltm.router.repository.update_ltm_job", new_callable=AsyncMock) as update_job,
            patch(
                "scrutator.ltm.router.repository.get_chunk_ids_by_source",
                new_callable=AsyncMock,
                return_value=["chunk-1", "chunk-2"],
            ),
            patch(
                "scrutator.search.indexer.index_document",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(chunks_indexed=2),
            ),
            patch("scrutator.ltm.router.IngestPipeline", return_value=pipeline),
            patch("scrutator.ltm.router._create_llm_client"),
            patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=pool),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/v1/ltm/ingest",
                json=BODY,
                headers={"X-LTM-Writer-Token": "ltm-secret"},
            )
    finally:
        settings.ltm_writer_token, settings.ltm_writer_namespaces = original

    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert update_job.await_args_list[-1].kwargs == {
        "status": "partial",
        "current_step": "enrichment_failed",
        "total_chunks": 2,
        "processed_chunks": 1,
        "error": "enrichment_failed",
    }


def test_job_status_endpoint_renders_partial_with_sanitized_error():
    job = {
        "id": "job-1",
        "namespace_id": 7,
        "source_path": "muneral://task/1",
        "status": "partial",
        "current_step": "enrichment_failed",
        "total_chunks": 2,
        "processed_chunks": 1,
        "error": "enrichment_failed",
    }
    with (
        override_tenant_context(app, READER),
        patch("scrutator.ltm.router.repository.get_ltm_job", new_callable=AsyncMock, return_value=job),
        TestClient(app) as client,
    ):
        response = client.get("/v1/ltm/jobs/job-1")

    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert response.json()["error"] == "enrichment_failed"
