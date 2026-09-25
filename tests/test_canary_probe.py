"""The canary producer, judged by the GATE'S OWN READER.

Nothing here asserts against a shape this repository invented. Every document `tools/canary_probe.py`
writes is handed to `canary_evidence` out of the vendored `.github/graph-admission/` bundle — the
same bytes `admit_change.canary_coverage` runs in CI — and the assertions are on what that reader
says. A fixture written by the same hand as the producer would agree with a producer that emits a
document the real consumer rejects (A2-287).

The contour is a stub HTTP server on a port the OS hands out, never a shared one and never
production: what is under test is the producer and the evidence binding, not the service.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / ".github" / "graph-admission" / "tools" / "graph"
PLAN = ROOT / "deploy" / "canary" / "scrutator-production.plan.json"


def _canary_probe():
    spec = importlib.util.spec_from_file_location("canary_probe", ROOT / "tools" / "canary_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canary_evidence():
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    import canary_evidence  # noqa: PLC0415

    return canary_evidence


class _Stub(BaseHTTPRequestHandler):
    edges_status = 403

    def log_message(self, *args):  # keep the test output readable
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "service": "scrutator", "version": "9.9.9"})
        elif self.path == "/v1/namespaces":
            self._send(200 if self.headers.get("Authorization") else 401, [])
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._send(type(self).edges_status, {"detail": "forbidden"})


@pytest.fixture
def contour():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def repo(tmp_path):
    """A one-commit repository standing in for the candidate."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "canary@example.invalid")
    git("config", "user.name", "canary")
    git("add", "-A")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "subject")
    return root


def _plan(**overrides):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan.update(overrides)
    return plan


def _run(module, plan, base_url, repo, out, **kwargs):
    return module.run(plan, base_url, "post", repo, out, 5.0, **kwargs)


def test_the_gates_own_reader_accepts_the_result_and_lists_the_routes(contour, repo, monkeypatch):
    module, evidence = _canary_probe(), _canary_evidence()
    monkeypatch.setenv("SCRUTATOR_CANARY_TOKEN", "not-a-real-token")
    out = repo / "receipts" / "canary" / "post.json"
    document = _run(module, _plan(), contour, repo, out)

    assert evidence.structure_errors(document) == []
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    rows, errors, _ = evidence.consume(str(out), repo, head)
    assert errors == []
    assert {r for r in rows} >= {"route:GET /health", "route:POST /v1/edges", "route:GET /v1/namespaces"}
    assert all(row["verdict"] == "verified" for row in rows.values()), rows
    assert document["resident_version"] == "9.9.9"


def test_a_regressed_403_turns_the_entity_red(contour, repo, monkeypatch):
    """Prove the check can go red: the A2-308 hole reopened is a `failed` route."""
    module, evidence = _canary_probe(), _canary_evidence()
    monkeypatch.setenv("SCRUTATOR_CANARY_TOKEN", "not-a-real-token")
    monkeypatch.setattr(_Stub, "edges_status", 200)
    document = _run(module, _plan(), contour, repo, repo / "receipts" / "canary" / "red.json")
    assert evidence.structure_errors(document) == []
    row = next(r for r in document["entity_verdicts"] if r["entity"] == "route:POST /v1/edges")
    assert row["verdict"] == "failed"
    assert "403" in row["reason"]


def test_a_5xx_is_not_measured_not_a_served_route(contour, repo, monkeypatch):
    """A fault is reachable, which is not the same as observed. Never a pass, never a regression."""
    module = _canary_probe()
    monkeypatch.setattr(_Stub, "edges_status", 503)
    plan = _plan()
    plan["probes"] = [
        {
            "id": "p",
            "method": "POST",
            "path": "/v1/edges",
            "mutating": True,
            "auth": "none",
            "expect": {"route_present": True},
            "entities": ["route:POST /v1/edges"],
        }
    ]
    document = _run(module, plan, contour, repo, repo / "receipts" / "canary" / "fault.json")
    row = document["entity_verdicts"][0]
    assert row["verdict"] == "not_measured" and "503" in row["reason"]


def test_a_renamed_route_is_a_failed_entity(contour, repo):
    """404 is the mutant class the offline verifiers cannot see."""
    module = _canary_probe()
    plan = _plan()
    plan["probes"] = [
        {
            "id": "p",
            "method": "GET",
            "path": "/healthz",
            "auth": "none",
            "expect": {"route_present": True},
            "entities": ["route:GET /health"],
        }
    ]
    document = _run(module, plan, contour, repo, repo / "receipts" / "canary" / "renamed.json")
    row = document["entity_verdicts"][0]
    assert row["verdict"] == "failed" and "404" in row["reason"]


def test_an_unreachable_contour_is_not_measured_never_verified(repo):
    module, evidence = _canary_probe(), _canary_evidence()
    document = _run(module, _plan(), "http://127.0.0.1:9", repo, repo / "receipts" / "canary" / "down.json")
    assert evidence.structure_errors(document) == []
    assert {r["verdict"] for r in document["entity_verdicts"]} == {"not_measured"}


def test_a_missing_credential_is_not_measured_never_failed(contour, repo, monkeypatch):
    module = _canary_probe()
    monkeypatch.delenv("SCRUTATOR_CANARY_TOKEN", raising=False)
    document = _run(module, _plan(), contour, repo, repo / "receipts" / "canary" / "nocred.json")
    row = next(r for r in document["entity_verdicts"] if r["entity"] == "route:GET /v1/namespaces")
    assert row["verdict"] == "not_measured"
    probe = next(p for p in document["probes"] if p["id"] == "namespaces-read")
    assert probe["executed"] is False


def test_an_undeclared_mutating_probe_refuses(contour, repo):
    module = _canary_probe()
    plan = _plan()
    for spec in plan["probes"]:
        spec.pop("mutating", None)
    with pytest.raises(module.Refusal, match="MUTATING_PROBE_UNDECLARED"):
        _run(module, plan, contour, repo, repo / "receipts" / "canary" / "c4.json")


def test_a_dirty_tree_has_no_immutable_subject(contour, repo):
    module = _canary_probe()
    (repo / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(module.Refusal, match="dirty"):
        _run(module, _plan(), contour, repo, repo / "receipts" / "canary" / "dirty.json")


def test_the_result_binds_after_being_committed_as_a_record_commit(contour, repo, monkeypatch):
    """The fixed point DEC-AUP-0040 R1 exists for, measured end to end.

    A commit cannot contain its own object id, so a result pinned to the candidate head could never
    be committed. R1 lets the measured subject be an ANCESTOR when the whole delta is record
    documents. This is that path: measure at the code commit, commit the evidence under receipts/,
    and read it back from GIT OBJECTS at the new head — which is what the gate does.
    """
    module, evidence = _canary_probe(), _canary_evidence()
    monkeypatch.setenv("SCRUTATOR_CANARY_TOKEN", "not-a-real-token")
    rel = "receipts/canary/post.json"
    measured = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    _run(module, _plan(), contour, repo, repo / rel)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("add", "-A")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "record: canary evidence")
    head = git("rev-parse", "HEAD")
    assert head != measured

    rows, errors, _ = evidence.consume(None, repo, head, at_head=rel)
    assert errors == [], errors
    assert "route:POST /v1/edges" in rows

    # …and a record commit that touches anything but a record document does NOT bind.
    (repo / "src" / "app.py").write_text("x = 3\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "not a record commit")
    _, errors, _ = evidence.consume(None, repo, git("rev-parse", "HEAD"), at_head=rel)
    assert errors and "src/app.py" in errors[0]


# ── A2-324: presence is not read off the status code ─────────────────────────────────────────
#
# The first production run of the route-presence plan went red on `GET /v1/ltm/jobs/{job_id}`:
# the handler answered 404 "Job not found" for the placeholder id, and the probe called a served
# route missing. These tests serve the REAL application — its own router, its own OpenAPI, its own
# no-match answer — so the probe is judged against what the resident version actually emits, and
# the mutant is the real one: the route deleted from the application.

ROUTE_PLAN = ROOT / "deploy" / "canary" / "scrutator-route-presence.plan.json"
TEMPLATED = ("get-v1-ltm-jobs-job-id", "get-v1-edges-chunk-id")


@pytest.fixture
def real_contour():
    """The application served by uvicorn on an OS-assigned port, with the database reads doubled
    as "absent" and a tenant context injected: what is under test is dispatch, not data."""
    import socket
    from unittest.mock import AsyncMock, patch

    import uvicorn

    from scrutator.health import app
    from tests.conftest import override_tenant_context

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, lifespan="off", log_level="error")
    server = uvicorn.Server(config)
    with (
        patch("scrutator.ltm.router.repository.get_ltm_job", AsyncMock(return_value=None)),
        patch("scrutator.health.get_edges_for_chunk", AsyncMock(return_value=[])),
        override_tenant_context(app),
    ):
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        for _ in range(200):
            if server.started:
                break
            threading.Event().wait(0.02)
        assert server.started, "uvicorn did not start"
        app.openapi_schema = None
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", app
        server.should_exit = True
        thread.join(5)
        app.openapi_schema = None
    sock.close()


def _route_plan(ids):
    plan = json.loads(ROUTE_PLAN.read_text(encoding="utf-8"))
    plan["probes"] = [p for p in plan["probes"] if p["id"] in ids]
    assert {p["id"] for p in plan["probes"]} == set(ids)
    return plan


def test_a_handlers_404_on_a_served_templated_route_is_verified(real_contour, repo):
    """The production false red, reproduced and closed: 404 "Job not found" is a served route."""
    base, _ = real_contour
    module, evidence = _canary_probe(), _canary_evidence()
    document = _run(module, _route_plan(TEMPLATED), base, repo, repo / "receipts" / "canary" / "tpl.json")
    assert evidence.structure_errors(document) == []
    jobs = next(p for p in document["probes"] if p["id"] == "get-v1-ltm-jobs-job-id")
    assert jobs["status"] == 404, "the fixture must reproduce the production observation"
    assert jobs["outcome"] == "verified", jobs["reason"]
    assert jobs["declared_by_resident_openapi"] is True
    assert document["resident_openapi"]["status"] == 200
    assert {r["verdict"] for r in document["entity_verdicts"]} == {"verified"}


def _serve_mutant(tmp_path, stale_openapi: bool = False):
    """The application with `@router.get("/jobs/{job_id}")` DELETED FROM THE SOURCE, served by its
    own uvicorn process. FastAPI 0.141 copies an included router's routes into a cached dispatch
    table, so removing a route from a live object is not the mutant — deleting it from the code is.
    With `stale_openapi` the mutant also serves the UNMUTATED OpenAPI document: a lying document."""
    import shutil
    import socket
    import time
    import urllib.request

    work = tmp_path / "mutant"
    shutil.copytree(ROOT / "src", work / "src")
    router = work / "src" / "scrutator" / "ltm" / "router.py"
    text = router.read_text(encoding="utf-8")
    decorator = '@router.get("/jobs/{job_id}", response_model=LtmJob)\n'
    assert text.count(decorator) == 1, "the mutant must remove exactly one route"
    router.write_text(text.replace(decorator, ""), encoding="utf-8")
    if stale_openapi:
        from scrutator.health import app

        (work / "openapi.json").write_text(json.dumps(app.openapi()), encoding="utf-8")
        health = work / "src" / "scrutator" / "health.py"
        health.write_text(
            health.read_text(encoding="utf-8")
            + f"\nimport json as _j\napp.openapi = lambda: _j.loads(open({str(work / 'openapi.json')!r}).read())\n",
            encoding="utf-8",
        )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    import os

    # Inherit the interpreter's environment (CI installs into a site the bare PATH does not reach)
    # and point the database at nothing: the mutant must never reach a Postgres on this host.
    env = {
        **os.environ,
        "PYTHONPATH": str(work / "src"),
        "SCRUTATOR_DATABASE_URL": "postgresql://none:none@127.0.0.1:1/none",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scrutator.health:app",
            "--port",
            str(port),
            "--lifespan",
            "off",
            "--log-level",
            "error",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            urllib.request.urlopen(base + "/health", timeout=1).read()
            return base, process
        except Exception:  # noqa: BLE001 — still starting
            if process.poll() is not None:
                break
            time.sleep(0.05)
    process.kill()
    pytest.fail(f"the mutant did not start: {process.stderr.read().decode()[-500:]}")


def _stop(process):
    process.terminate()
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()


def test_the_route_deleted_from_the_application_is_failed(repo, tmp_path):
    """The red control the card requires: the same probe against the application without the route."""
    module = _canary_probe()
    base, process = _serve_mutant(tmp_path)
    try:
        document = _run(module, _route_plan(TEMPLATED), base, repo, repo / "receipts" / "canary" / "mut.json")
    finally:
        _stop(process)
    jobs = next(p for p in document["probes"] if p["id"] == "get-v1-ltm-jobs-job-id")
    assert jobs["status"] == 404
    assert jobs["declared_by_resident_openapi"] is False
    assert jobs["outcome"] == "failed" and "does not declare" in jobs["reason"], jobs["reason"]
    row = next(r for r in document["entity_verdicts"] if r["entity"] == "route:GET /jobs/{job_id}")
    assert row["verdict"] == "failed"
    # The sibling is still declared; with no database behind it it answers a fault, which is
    # not_measured — the mutant is localised, not a blanket red.
    edges = next(p for p in document["probes"] if p["id"] == "get-v1-edges-chunk-id")
    assert edges["declared_by_resident_openapi"] is True and edges["outcome"] != "failed", edges


def test_the_routers_own_404_is_failed_even_when_the_openapi_still_lists_the_route(repo, tmp_path):
    """The second observation holds alone: an OpenAPI document that lies (stale, hand-written)
    cannot turn the router's no-match answer green."""
    module = _canary_probe()
    base, process = _serve_mutant(tmp_path, stale_openapi=True)
    try:
        document = _run(module, _route_plan(TEMPLATED), base, repo, repo / "receipts" / "canary" / "lie.json")
    finally:
        _stop(process)
    jobs = next(p for p in document["probes"] if p["id"] == "get-v1-ltm-jobs-job-id")
    assert jobs["declared_by_resident_openapi"] is True, "the mutant must serve the lying document"
    assert jobs["outcome"] == "failed" and "no-match" in jobs["reason"], jobs["reason"]


class _VerbStub(_Stub):
    """A contour whose path exists but whose verb was rewritten, and whose OpenAPI is unreadable."""

    def do_GET(self):
        if self.path == "/v1/thing/00000000-0000-0000-0000-000000000000":
            self._send(404, {"detail": "thing not found"})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        self._send(405, {"detail": "Method Not Allowed"})


@pytest.fixture
def verb_contour():
    server = HTTPServer(("127.0.0.1", 0), _VerbStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _one(spec):
    plan = _plan()
    plan["probes"] = [{"auth": "none", "expect": {"route_present": True}, "entities": ["route:X"], **spec}]
    return plan


def test_a_405_is_a_rewritten_verb_not_a_served_route(verb_contour, repo):
    """Before A2-324 any status but 404 counted as served, so `@post` rewritten to `@put` was green."""
    module = _canary_probe()
    plan = _one({"id": "p", "method": "POST", "path": "/v1/thing", "mutating": True})
    document = _run(module, plan, verb_contour, repo, repo / "receipts" / "canary" / "405.json")
    assert document["probes"][0]["outcome"] == "failed" and "405" in document["probes"][0]["reason"]


def test_a_404_that_nothing_can_attribute_is_not_measured(verb_contour, repo):
    """OpenAPI unreadable, body differs from the no-match fingerprint: absence of measurement (C3),
    neither the old false red nor a pass."""
    module = _canary_probe()
    plan = _one({"id": "p", "method": "GET", "path": "/v1/thing/00000000-0000-0000-0000-000000000000"})
    document = _run(module, plan, verb_contour, repo, repo / "receipts" / "canary" / "unattr.json")
    probe = document["probes"][0]
    assert probe["declared_by_resident_openapi"] is None
    assert probe["outcome"] == "not_measured", probe["reason"]
