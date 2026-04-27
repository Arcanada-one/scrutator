"""Tests for LTM-0013 router endpoints — POST /reflect, GET /meta_facts."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.ltm.models import FactType, MetaFact, ReflectRunSummary


def _client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


def _summary(**over):
    base = dict(
        run_id="r-1",
        status="done",
        chunks_scanned=2,
        meta_facts_created=1,
        cost_usd=0.0,
        req_count=1,
        duration_ms=100.0,
    )
    base.update(over)
    return ReflectRunSummary(**base)


def _fact(**over):
    base = dict(
        namespace="arcanada",
        fact_type=FactType.SUMMARY,
        content="Sample meta-fact",
        source_chunk_ids=["00000000-0000-0000-0000-000000000001"],
        model_used="openrouter/gemini-2.5-flash",
    )
    base.update(over)
    return MetaFact(**base)


class TestReflectEndpoint:
    def test_reflect_dry_run_returns_preview(self):
        with (
            patch(
                "scrutator.ltm.router.repository.upsert_namespace",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch("scrutator.ltm.router.ReflectJob") as MockJob,
        ):
            instance = MockJob.return_value
            instance.run = AsyncMock(return_value=(_summary(), [_fact()]))
            resp = _client().post("/v1/ltm/reflect", json={"namespace": "arcanada", "dry_run": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["preview"] is not None
        assert len(body["preview"]) == 1
        assert body["summary"]["status"] == "done"

    def test_reflect_persist_omits_preview(self):
        with (
            patch(
                "scrutator.ltm.router.repository.upsert_namespace",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch("scrutator.ltm.router.ReflectJob") as MockJob,
        ):
            instance = MockJob.return_value
            instance.run = AsyncMock(return_value=(_summary(), [_fact()]))
            resp = _client().post("/v1/ltm/reflect", json={"namespace": "arcanada"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["preview"] is None

    def test_reflect_disabled_returns_503(self):
        original = settings.ltm_reflect_enabled
        settings.ltm_reflect_enabled = False
        try:
            resp = _client().post("/v1/ltm/reflect", json={"namespace": "arcanada"})
            assert resp.status_code == 503
        finally:
            settings.ltm_reflect_enabled = original


class TestMetaFactsEndpoint:
    def test_list_meta_facts_filters_by_type(self):
        sample = [
            {
                "id": "mf-1",
                "fact_type": "summary",
                "content": "x",
                "source_chunk_ids": [],
                "entity_ids": [],
                "depth": 1,
                "derived_at": None,
                "model_used": "m",
                "reflect_run_id": None,
                "properties": {},
            }
        ]
        with (
            patch(
                "scrutator.ltm.router.repository.upsert_namespace",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch(
                "scrutator.ltm.router.repository.list_meta_facts_by_namespace",
                new_callable=AsyncMock,
                return_value=sample,
            ) as mock_list,
        ):
            resp = _client().get("/v1/ltm/meta_facts", params={"fact_type": "summary"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["meta_facts"][0]["fact_type"] == "summary"
        mock_list.assert_awaited_once()
        kwargs = mock_list.call_args.kwargs
        assert kwargs["fact_type"] == "summary"

    def test_list_meta_facts_caps_limit(self):
        with (
            patch(
                "scrutator.ltm.router.repository.upsert_namespace",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch(
                "scrutator.ltm.router.repository.list_meta_facts_by_namespace",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_list,
        ):
            resp = _client().get("/v1/ltm/meta_facts", params={"limit": 9999})
        assert resp.status_code == 200
        kwargs = mock_list.call_args.kwargs
        assert kwargs["limit"] == 500
