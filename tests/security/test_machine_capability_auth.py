from unittest.mock import patch

from fastapi.testclient import TestClient

from scrutator.health import app


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_feeder_capability_is_evaluated_without_bearer_first():
    with (
        patch("scrutator.auth.dependency.settings") as reader_settings,
        patch("scrutator.auth.capabilities.settings") as capability_settings,
    ):
        reader_settings.auth_enforce = True
        capability_settings.feeder_token = "feeder-secret"
        capability_settings.feeder_namespaces = "wiki"
        response = _client().post(
            "/v1/index",
            json={},
            headers={"X-KB-Feeder-Token": "feeder-secret"},
        )
    assert response.status_code == 422


def test_rollback_capability_is_evaluated_without_bearer_first():
    with (
        patch("scrutator.auth.dependency.settings") as reader_settings,
        patch("scrutator.auth.capabilities.settings") as capability_settings,
    ):
        reader_settings.auth_enforce = True
        capability_settings.rollback_token = "rollback-secret"
        capability_settings.rollback_namespaces = "wiki"
        capability_settings.operator_rollback_token = "operator-secret"
        response = _client().request(
            "DELETE",
            "/v1/index",
            json={},
            headers={"X-KB-Rollback-Token": "rollback-secret"},
        )
    assert response.status_code == 422


def test_ltm_writer_capability_is_evaluated_without_bearer_first():
    with (
        patch("scrutator.auth.dependency.settings") as reader_settings,
        patch("scrutator.auth.capabilities.settings") as capability_settings,
    ):
        reader_settings.auth_enforce = True
        capability_settings.ltm_writer_token = "writer-secret"
        capability_settings.ltm_writer_namespaces = "muneral"
        capability_settings.ltm_writer_source_prefixes = '{"muneral":["muneral://task/"]}'
        for method, path in (("POST", "/v1/ltm/ingest"), ("DELETE", "/v1/ltm/source")):
            response = _client().request(
                method,
                path,
                json={},
                headers={"X-LTM-Writer-Token": "writer-secret"},
            )
            assert response.status_code == 422


def test_machine_routes_reject_wrong_capability_even_in_grace_mode():
    with patch("scrutator.auth.capabilities.settings") as capability_settings:
        capability_settings.feeder_token = "feeder-secret"
        capability_settings.feeder_namespaces = "wiki"
        response = _client().post(
            "/v1/index",
            json={},
            headers={"X-KB-Feeder-Token": "wrong"},
        )
    assert response.status_code == 401
