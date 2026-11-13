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


PROXY_PEER = ("172.23.0.4", 50000)


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

    @app.get("/v1/cap-read")
    async def cap_read(cap=Depends(capabilities.require_feeder_capability)):
        return {"namespaces": sorted(cap.namespaces)}

    @app.delete("/v1/cap-delete")
    async def cap_delete(cap=Depends(capabilities.require_feeder_capability)):
        return {"deleted": True}

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


def _rec(
    ts: datetime, identity: str, *, mode="read", ns=(), status=200, src="127.0.0.1", auth="ok", ua=None, fwd=None
) -> dict:
    return {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "identity": identity,
        "auth": auth,
        "mode": mode,
        "src": src,
        "fwd": fwd,
        "ua": ua,
        "method": "POST",
        "route": "/v1/x",
        "namespaces": list(ns),
        "allowed_n": None,
        "status": status,
        "ms": 5,
        "qs_len": 0,
    }


def _groups(report, identity):
    return [g for g in report["groups"] if g["identity"] == identity]


def test_the_report_groups_by_identity_ua_and_proxy_reported_client(tmp_path, monkeypatch):
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _write(
        tmp_path,
        [
            _rec(now - timedelta(hours=2), "reader:agent-a", ns=["alpha"], ua="claude-code/2.1", fwd="100.64.0.7"),
            _rec(
                now - timedelta(hours=1), "reader:agent-a", ns=["alpha", "beta"], ua="claude-code/2.1", fwd="100.64.0.7"
            ),
            _rec(
                now - timedelta(hours=1), "reader:agent-a", ua="codex/0.9", fwd="100.64.0.8"
            ),  # same principal, other agent
            _rec(now - timedelta(minutes=30), "capability:feeder", mode="write", ns=["alpha"], ua="python-httpx/0.27"),
            _rec(
                now - timedelta(minutes=5), "capability:feeder", mode="write", ns=["alpha"], status=401, auth="denied"
            ),
            _rec(
                now - timedelta(days=3), "capability:ltm-writer", mode="write", ns=["gamma"]
            ),  # outside a 1-day window
            _rec(now - timedelta(minutes=1), "none", auth="none", status=401),
        ],
    )
    monkeypatch.setattr(
        access_report,
        "database_lists",
        lambda use_db: (
            ["reader:agent-a", "reader:idle-agent"],
            [("alpha", True), ("beta", True), ("gamma", False), ("delta", True)],
            "test",
        ),
    )
    report = access_report.build_report(tmp_path, 1, use_db=True, now=now)
    assert report["records"] == 6
    a_groups = _groups(report, "reader:agent-a")
    assert {(g["ua"], g["fwd"]) for g in a_groups} == {("claude-code/2.1", "100.64.0.7"), ("codex/0.9", "100.64.0.8")}
    claude = next(g for g in a_groups if g["ua"] == "claude-code/2.1")
    assert (claude["requests"], claude["reads"], claude["writes"]) == (2, 2, 0)
    assert claude["namespaces"] == {"alpha": 2, "beta": 1}
    feeder = _groups(report, "capability:feeder")
    assert sorted((g["ua"], g["denied"]) for g in feeder) == [("-", 1), ("python-httpx/0.27", 0)]
    assert next(g for g in feeder if g["ua"] == "-")["fwd"] == "-"
    assert report["identity_totals"]["reader:agent-a"] == 3
    assert _groups(report, "none")[0]["denied"] == 1
    assert (
        "capability:ltm-writer" in report["zero_use_identities"]
        and "reader:idle-agent" in report["zero_use_identities"]
    )
    assert "reader:agent-a" not in report["zero_use_identities"]
    # delta was never read in the window, gamma neither; gamma also has no reader grant at all
    assert report["namespaces_without_reads_in_window"] == ["delta", "gamma"]
    assert report["namespaces_without_any_reader_grant"] == ["gamma"]
    text = access_report.render_text(report)
    assert "ua=claude-code/2.1  fwd=100.64.0.7" in text and "NO reader grant" in text and "Limits:" in text
    assert "not verified" in text


def test_a_write_or_a_denied_request_does_not_count_as_a_read_of_a_namespace(tmp_path, monkeypatch):
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _write(
        tmp_path,
        [
            _rec(now - timedelta(minutes=3), "capability:feeder", mode="write", ns=["alpha"]),
            _rec(now - timedelta(minutes=2), "reader:agent-a", ns=["beta"], status=403),
            _rec(now - timedelta(minutes=1), "reader:agent-a", ns=[]),  # unscoped read names no namespace
        ],
    )
    monkeypatch.setattr(
        access_report, "database_lists", lambda use_db: (None, [("alpha", True), ("beta", True)], "test")
    )
    report = access_report.build_report(tmp_path, 1, use_db=True, now=now)
    assert report["namespaces_without_reads_in_window"] == ["alpha", "beta"]


def test_without_a_database_the_namespace_lists_are_reported_as_unavailable(tmp_path):
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _write(tmp_path, [_rec(now - timedelta(minutes=1), "reader:agent-a", ns=["alpha"])])
    report = access_report.build_report(tmp_path, 1, use_db=False, now=now)
    assert (
        report["namespaces_without_reads_in_window"] is None and report["namespaces_without_any_reader_grant"] is None
    )
    assert "not available" in access_report.render_text(report)


def test_the_report_skips_corrupt_lines_and_reads_rotated_files(tmp_path):
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    (tmp_path / (access_log.ACCESS_FILE + ".2026-11-13")).write_text(
        json.dumps(_rec(now - timedelta(hours=20), "reader:agent-b", ns=["beta"])) + "\nnot json at all\n",
        encoding="utf-8",
    )
    _write(tmp_path, [_rec(now - timedelta(minutes=2), "reader:agent-b", ns=["beta"])])
    report = access_report.build_report(tmp_path, 2, use_db=False, now=now)
    assert _groups(report, "reader:agent-b")[0]["requests"] == 2


# --- R1: the two attribution fields that are not credentials ------------------------------------------


def _client_with(app, peer):
    return TestClient(app, raise_server_exceptions=False, client=peer)


def test_ua_and_the_proxy_reported_address_are_recorded_when_the_peer_is_the_proxy(client_and_dir):
    client, directory = client_and_dir
    app = client.app
    with _client_with(app, PROXY_PEER) as proxied:
        proxied.get(
            "/v1/open", headers={"User-Agent": "claude-code/2.1 (cli)", "X-Forwarded-For": "100.64.0.7, 172.23.0.4"}
        )
    (record,) = _records(directory)
    assert record["src"] == "172.23.0.4"
    assert record["fwd"] == "100.64.0.7"  # first hop of X-Forwarded-For
    assert record["ua"] == "claude-code/2.1 (cli)"


def test_cf_connecting_ip_wins_over_x_forwarded_for(client_and_dir):
    client, directory = client_and_dir
    with _client_with(client.app, PROXY_PEER) as proxied:
        proxied.get("/v1/open", headers={"CF-Connecting-IP": "203.0.113.9", "X-Forwarded-For": "100.64.0.7"})
    assert _records(directory)[0]["fwd"] == "203.0.113.9"


def test_a_forwarded_address_from_a_public_peer_is_ignored_so_it_cannot_be_forged(client_and_dir):
    client, directory = client_and_dir
    for peer in (("8.8.8.8", 40000), ("100.101.102.103", 40000)):  # a public caller, and a direct mesh caller
        with _client_with(client.app, peer) as remote:
            remote.get("/v1/open", headers={"X-Forwarded-For": "10.1.2.3", "CF-Connecting-IP": "10.1.2.3"})
    public, mesh = _records(directory)
    assert (public["src"], public["fwd"]) == ("8.8.8.8", None)
    assert (mesh["src"], mesh["fwd"]) == ("100.101.102.103", None)  # its own address is already in src


@pytest.mark.parametrize("forwarded", ["not-an-address", "100.64.0.7; DROP", "999.1.1.1", "", " , "])
def test_an_invalid_forwarded_address_is_recorded_as_null(client_and_dir, forwarded):
    client, directory = client_and_dir
    with _client_with(client.app, ("127.0.0.1", 40000)) as local:
        local.get("/v1/open", headers={"X-Forwarded-For": forwarded})
    assert _records(directory)[0]["fwd"] is None


def test_the_user_agent_is_cut_to_80_characters_and_made_printable_ascii(client_and_dir):
    client, directory = client_and_dir
    client.get("/v1/open", headers={"User-Agent": "agent/1.0 " + "x" * 200})
    client.get("/v1/open", headers={"User-Agent": "caf\xe9\x01\x7f-tool".encode("latin-1")})
    client.get("/v1/open", headers={"User-Agent": ""})
    long_ua, odd_ua, empty_ua = [r["ua"] for r in _records(directory)]
    assert len(long_ua) == 80 and long_ua.startswith("agent/1.0 xxx")
    assert odd_ua.startswith("caf") and odd_ua.endswith("-tool")
    assert all(" " <= ch <= "~" for ch in odd_ua), "every non-printable or non-ASCII byte is replaced"
    assert "?" in odd_ua and "\x01" not in odd_ua and "\x7f" not in odd_ua
    assert empty_ua is None


def test_no_credential_header_value_ever_reaches_a_record_even_with_ua_and_fwd_recorded(client_and_dir):
    client, directory = client_and_dir
    headers = {
        "Authorization": f"Bearer {SECRET}",
        "X-KB-Feeder-Token": SECRET,
        "X-LTM-Writer-Token": SECRET + "-ltm",
        "X-KB-Rollback-Token": SECRET + "-rb",
        "X-Capability-Projection-Token": SECRET + "-cp",
        "Cookie": "session=" + SECRET,
        "User-Agent": "tool/1.0",
        "X-Forwarded-For": "100.64.0.7",
    }
    with _client_with(client.app, PROXY_PEER) as proxied:
        proxied.post("/v1/write", headers=headers)
        proxied.get("/v1/open", headers=headers)
    raw = (directory / access_log.ACCESS_FILE).read_text(encoding="utf-8")
    assert SECRET not in raw and "Bearer" not in raw and "session=" not in raw
    assert all(r["ua"] == "tool/1.0" and r["fwd"] == "100.64.0.7" for r in _records(directory))


def test_the_middleware_reads_only_three_named_request_headers():
    scope = {
        "headers": [(b"authorization", b"A"), (b"x-kb-feeder-token", b"B"), (b"user-agent", b"U"), (b"cookie", b"C")],
        "client": ("127.0.0.1", 1),
    }
    asked: list[bytes] = []
    original = access_log._header

    def spy(s, name):
        asked.append(name)
        return original(s, name)

    access_log._header, saved = spy, access_log._header
    try:
        access_log.client_agent(scope)
        access_log.reported_client(scope)
    finally:
        access_log._header = saved
    assert set(asked) <= {b"user-agent", b"cf-connecting-ip", b"x-forwarded-for"}


# --- R2: the single-process assumption is visible -----------------------------------------------------


def test_the_single_process_assumption_is_logged_and_a_multi_worker_setting_warns(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="scrutator.access_log")
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    access_log.AccessLogMiddleware(lambda *a: None, directory=str(tmp_path), backups=3)
    assert any("single-process assumption" in r.getMessage() for r in caplog.records)
    assert not any(r.levelname == "WARNING" for r in caplog.records)
    caplog.clear()
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    access_log.AccessLogMiddleware(lambda *a: None, directory=str(tmp_path), backups=3)
    assert any(r.levelname == "WARNING" and "4 workers" in r.getMessage() for r in caplog.records)


# --- mode follows the verb for machine capabilities (control decision on 4e0054f0) ---------------------


def test_a_capability_get_is_logged_as_a_read_and_a_capability_post_or_delete_as_a_write(client_and_dir):
    client, directory = client_and_dir
    headers = {"X-KB-Feeder-Token": SECRET}
    assert client.get("/v1/cap-read", headers=headers).status_code == 200
    assert client.post("/v1/write", headers=headers).status_code == 200
    assert client.delete("/v1/cap-delete", headers=headers).status_code == 200
    got, posted, deleted = _records(directory)
    assert (got["identity"], got["mode"]) == ("capability:feeder", "read")
    assert (posted["identity"], posted["mode"]) == ("capability:feeder", "write")
    assert (deleted["identity"], deleted["mode"]) == ("capability:feeder", "write")


def test_a_refused_capability_get_is_a_read_too_and_a_refused_post_stays_a_write(client_and_dir):
    client, directory = client_and_dir
    assert client.get("/v1/cap-read").status_code == 401
    assert client.post("/v1/write").status_code == 401
    refused_get, refused_post = _records(directory)
    assert (refused_get["mode"], refused_get["auth"], refused_get["status"]) == ("read", "none", 401)
    assert (refused_post["mode"], refused_post["auth"], refused_post["status"]) == ("write", "none", 401)


def test_only_the_logged_field_changes_the_capability_still_authorises_the_same_way(client_and_dir):
    client, _directory = client_and_dir
    assert client.get("/v1/cap-read", headers={"X-KB-Feeder-Token": "wrong"}).status_code == 401
    assert client.get("/v1/cap-read", headers={"X-KB-Feeder-Token": SECRET}).json() == {"namespaces": ["alpha", "beta"]}
