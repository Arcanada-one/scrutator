"""Durable access log (Muneral 4e0054f0): record content, retention, report, secrets, unauthenticated."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from scrutator import access_log
from scrutator.auth import capabilities
from scrutator.config import settings
from scrutator.tools import access_report

SECRET = "tok-SECRET-9f2c7a1d4b8e-do-not-log"
QUERY_TEXT = "what is the private pricing plan of project mercury"


def _records(directory: Path) -> list[dict]:
    path = directory / access_log.ACCESS_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def client_and_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "feeder_token", SECRET)
    monkeypatch.setattr(settings, "feeder_namespaces", "alpha,beta")
    app = FastAPI()
    app.add_middleware(access_log.AccessLogMiddleware, directory=str(tmp_path), backups=3)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/write")
    async def write(cap=Depends(capabilities.require_feeder_capability)):
        access_log.note_namespace("alpha")
        return {"namespaces": sorted(cap.namespaces)}

    @app.post("/v1/search")
    async def search():
        # what the real dependency and handler do for a reader
        access_log.note_identity("reader:svc-agent-a", auth="ok", mode="read")
        access_log.note_allowed(3)
        access_log.note_query(QUERY_TEXT)
        access_log.note_namespace("alpha")
        return {"results": []}

    @app.get("/v1/open")
    async def open_route():
        return {"x": 1}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("handler failure")

    return TestClient(app, raise_server_exceptions=False), tmp_path


def test_a_request_with_a_token_leaves_a_record_without_the_token_value(client_and_dir):
    client, directory = client_and_dir
    response = client.post(
        "/v1/write?extra=1",
        headers={"X-KB-Feeder-Token": SECRET, "Authorization": f"Bearer {SECRET}"},
    )
    assert response.status_code == 200
    raw = (directory / access_log.ACCESS_FILE).read_text(encoding="utf-8")
    assert SECRET not in raw, "a credential value reached the access log"
    (record,) = _records(directory)
    assert record["identity"] == "capability:feeder"
    assert record["auth"] == "ok" and record["mode"] == "write"
    assert record["method"] == "POST" and record["route"] == "/v1/write"
    assert record["namespaces"] == ["alpha"]
    assert record["status"] == 200 and isinstance(record["ms"], int)
    assert record["qs_len"] == len("extra=1")
    assert record["ts"].endswith("Z") and len(record["ts"]) == 24
    assert record["src"]


def test_a_refused_credential_is_denied_and_a_missing_one_is_none(client_and_dir):
    client, directory = client_and_dir
    assert client.post("/v1/write", headers={"X-KB-Feeder-Token": "wrong-" + SECRET}).status_code == 401
    assert client.post("/v1/write").status_code == 401
    assert (directory / access_log.ACCESS_FILE).read_text(encoding="utf-8").count(SECRET) == 0
    refused, missing = _records(directory)
    assert (refused["identity"], refused["auth"], refused["status"]) == ("capability:feeder", "denied", 401)
    assert (missing["identity"], missing["auth"], missing["status"]) == ("capability:feeder", "none", 401)


def test_an_unauthenticated_request_is_recorded_as_such(client_and_dir):
    client, directory = client_and_dir
    assert client.get("/v1/open").status_code == 200
    (record,) = _records(directory)
    assert (record["identity"], record["auth"], record["mode"]) == ("none", "none", "read")
    assert record["namespaces"] == [] and record["allowed_n"] is None


def test_a_query_is_stored_as_a_length_and_a_hash_only(client_and_dir):
    client, directory = client_and_dir
    client.post("/v1/search", json={"query": QUERY_TEXT}, headers={"Authorization": f"Bearer {SECRET}"})
    raw = (directory / access_log.ACCESS_FILE).read_text(encoding="utf-8")
    assert QUERY_TEXT not in raw and "mercury" not in raw and SECRET not in raw
    (record,) = _records(directory)
    assert record["identity"] == "reader:svc-agent-a" and record["mode"] == "read" and record["allowed_n"] == 3
    assert record["q_len"] == len(QUERY_TEXT) and len(record["q_sha16"]) == 16
    assert record["namespaces"] == ["alpha"]


def test_health_is_not_recorded_and_an_unmatched_route_is(client_and_dir):
    client, directory = client_and_dir
    assert client.get("/health").status_code == 200
    assert _records(directory) == []
    assert client.get("/no/such/route").status_code == 404
    (record,) = _records(directory)
    assert record["route"] == "unmatched" and record["status"] == 404


def test_a_failing_handler_is_still_recorded_as_500(client_and_dir):
    client, directory = client_and_dir
    assert client.get("/boom").status_code == 500
    (record,) = _records(directory)
    assert record["route"] == "/boom" and record["status"] == 500


def test_a_broken_log_directory_never_breaks_a_request(tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    app = FastAPI()
    app.add_middleware(access_log.AccessLogMiddleware, directory=str(blocker / "sub"), backups=3)

    @app.get("/v1/open")
    async def open_route():
        return {"x": 1}

    assert TestClient(app).get("/v1/open").json() == {"x": 1}


def test_an_empty_directory_setting_turns_the_log_off(tmp_path):
    app = FastAPI()
    app.add_middleware(access_log.AccessLogMiddleware, directory=None, backups=3)

    @app.get("/v1/open")
    async def open_route():
        return {"x": 1}

    assert TestClient(app).get("/v1/open").status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_retention_is_more_than_thirty_days():
    assert settings.access_log_backups >= 30
    assert access_log.DEFAULT_BACKUPS >= 30


def test_the_real_capability_dependencies_never_pass_a_value_to_the_log(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(access_log, "note_identity", lambda *a, **k: seen.append((a, k)))
    monkeypatch.setattr(settings, "ltm_writer_token", SECRET)
    import asyncio

    asyncio.run(capabilities.require_ltm_writer_capability(SECRET))
    with pytest.raises(HTTPException):
        asyncio.run(capabilities.require_ltm_writer_capability("not-" + SECRET))
    assert seen and all(SECRET not in repr(call) for call in seen)
    assert seen[0][0][0] == "capability:ltm-writer" and seen[0][1]["auth"] == "ok"
    assert seen[1][1]["auth"] == "denied"


def _write(directory: Path, lines: list[dict]) -> None:
    (directory / access_log.ACCESS_FILE).write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


def _rec(ts: datetime, identity: str, *, mode="read", ns=(), status=200, src="127.0.0.1", auth="ok") -> dict:
    return {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "identity": identity,
        "auth": auth,
        "mode": mode,
        "src": src,
        "method": "POST",
        "route": "/v1/x",
        "namespaces": list(ns),
        "allowed_n": None,
        "status": status,
        "ms": 5,
        "qs_len": 0,
    }


def test_the_report_groups_by_identity_and_lists_zero_use(tmp_path):
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _write(
        tmp_path,
        [
            _rec(now - timedelta(hours=2), "reader:agent-a", ns=["alpha"]),
            _rec(now - timedelta(hours=1), "reader:agent-a", ns=["alpha", "beta"]),
            _rec(now - timedelta(hours=1), "reader:agent-a"),
            _rec(now - timedelta(minutes=30), "capability:feeder", mode="write", ns=["alpha"], src="100.64.0.9"),
            _rec(
                now - timedelta(minutes=5), "capability:feeder", mode="write", ns=["alpha"], status=401, auth="denied"
            ),
            _rec(
                now - timedelta(days=3), "capability:ltm-writer", mode="write", ns=["gamma"]
            ),  # outside a 1-day window
            _rec(now - timedelta(minutes=1), "none", auth="none", status=401),
        ],
    )
    report = access_report.build_report(tmp_path, 1, use_db=False, now=now)
    assert report["records"] == 6
    a = report["identities"]["reader:agent-a"]
    assert (a["requests"], a["reads"], a["writes"]) == (3, 3, 0)
    assert a["namespaces"] == {"alpha": 2, "beta": 1, "(unscoped)": 1}
    f = report["identities"]["capability:feeder"]
    assert (f["writes"], f["denied"]) == (2, 1) and f["sources"] == {"100.64.0.9": 1, "127.0.0.1": 1}
    assert f["first_seen"] < f["last_seen"]
    assert report["identities"]["none"]["denied"] == 1
    assert "capability:ltm-writer" in report["zero_use"] and "capability:feeder" not in report["zero_use"]
    text = access_report.render_text(report)
    assert "ZERO use" in text and "reader:agent-a" in text and "Limits:" in text


def test_the_report_skips_corrupt_lines_and_reads_rotated_files(tmp_path):
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    (tmp_path / (access_log.ACCESS_FILE + ".2026-11-13")).write_text(
        json.dumps(_rec(now - timedelta(hours=20), "reader:agent-b", ns=["beta"])) + "\nnot json at all\n",
        encoding="utf-8",
    )
    _write(tmp_path, [_rec(now - timedelta(minutes=2), "reader:agent-b", ns=["beta"])])
    report = access_report.build_report(tmp_path, 2, use_db=False, now=now)
    assert report["identities"]["reader:agent-b"]["requests"] == 2
