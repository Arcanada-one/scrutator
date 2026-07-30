from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.health import app


def test_feeder_capability_requires_the_existing_machine_credential():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement,arcanada-design-system"
    try:
        with TestClient(app) as client:
            missing = client.get("/v1/index/capability")
            wrong = client.get(
                "/v1/index/capability",
                headers={"X-KB-Feeder-Token": "wrong-secret"},
            )
            accepted = client.get(
                "/v1/index/capability",
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {
        "schema_version": 1,
        "namespaces": ["arcanada-design-system", "self-improvement"],
    }
    assert "secret" not in accepted.text


def test_rollback_capability_requires_and_describes_the_actual_rollback_credential():
    original = (
        settings.rollback_token,
        settings.operator_rollback_token,
        settings.rollback_namespaces,
    )
    settings.rollback_token = "rollback-secret"
    settings.operator_rollback_token = "operator-secret"
    settings.rollback_namespaces = "wiki,arcanada-design-system"
    try:
        with TestClient(app) as client:
            missing = client.get("/v1/index/rollback-capability")
            scoped = client.get(
                "/v1/index/rollback-capability",
                headers={"X-KB-Rollback-Token": "rollback-secret"},
            )
            operator = client.get(
                "/v1/index/rollback-capability",
                headers={"X-KB-Rollback-Token": "operator-secret"},
            )
    finally:
        (
            settings.rollback_token,
            settings.operator_rollback_token,
            settings.rollback_namespaces,
        ) = original

    assert missing.status_code == 401
    assert scoped.json() == {
        "schema_version": 1,
        "namespaces": ["arcanada-design-system", "wiki"],
        "operator": False,
    }
    assert operator.json() == {
        "schema_version": 1,
        "namespaces": ["arcanada-design-system", "wiki"],
        "operator": True,
    }
