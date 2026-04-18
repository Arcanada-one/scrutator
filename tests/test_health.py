"""Smoke test for the health endpoint."""

from fastapi.testclient import TestClient

from scrutator.health import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Scrutator"
    assert "version" in data
