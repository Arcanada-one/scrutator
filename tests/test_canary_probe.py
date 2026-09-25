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
