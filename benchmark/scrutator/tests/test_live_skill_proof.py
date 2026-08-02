"""Contract tests for the loopback-only skills proof runner."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "live_skill_proof.py"
MODULE_SPEC = importlib.util.spec_from_file_location("live_skill_proof", SCRIPT)
assert MODULE_SPEC and MODULE_SPEC.loader
proof = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(proof)


class _ProbeHandler(BaseHTTPRequestHandler):
    responses: dict[tuple[str, str], tuple[int, object, str]] = {}
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def _respond(self, method: str) -> None:
        parsed = urlsplit(self.path)
        body: dict[str, object] | None = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw) if raw else None
        self.__class__.requests.append((method, self.path, body))

        status, payload, content_type = self.__class__.responses.get(
            (method, parsed.path),
            (404, {"detail": "not found"}, "application/json"),
        )
        encoded = (
            str(payload).encode("utf-8")
            if content_type == "text/plain"
            else json.dumps(payload).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        self._respond("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        self._respond("POST")

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def probe_server():
    doc_id = proof._PROBE_DOC_ID
    content = "x" * proof.PROBE_BYTE_COUNT
    _ProbeHandler.responses = {
        ("GET", "/health"): (200, {"service": "Scrutator", "version": "test"}, "application/json"),
        ("GET", "/v1/namespaces"): (200, [{"name": "skills"}], "application/json"),
        ("POST", "/v1/search"): (
            200,
            {
                "results": [
                    {
                        "source_path": proof.PROBE_SOURCE_PATH,
                        "source_id": doc_id,
                        "content_hash": proof.PROBE_CONTENT_HASH,
                        "namespace": proof.REQUIRED_NAMESPACE,
                    }
                ],
                "total": 1,
                "query": proof.PROBE_SEARCH_QUERY,
                "search_time_ms": 1.0,
            },
            "application/json",
        ),
        ("GET", "/v1/navigate/outline"): (
            200,
            {
                "source_path": proof.PROBE_SOURCE_PATH,
                "namespace": proof.REQUIRED_NAMESPACE,
                "doc_id": doc_id,
                "total_chunks": 1,
                "outline": [{"title": "probe", "anchor": "probe"}],
            },
            "application/json",
        ),
        ("POST", "/v1/fetch"): (
            200,
            {
                "content_hash": proof.PROBE_CONTENT_HASH,
                "content": content,
                "namespace": proof.REQUIRED_NAMESPACE,
                "trust_class": "skill",
                "content_exact": True,
                "path": proof.PROBE_SOURCE_PATH,
            },
            "application/json",
        ),
    }
    _ProbeHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProbeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _ProbeHandler.requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_dry_run_checks_registry_without_network(tmp_path):
    assert proof.main(
        [
            "--endpoint",
            "http://127.0.0.1:1",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    ) == proof.SUCCESS_CODE
    assert not (tmp_path / "summary.json").exists()


def test_valid_proof_requires_search_discovery_and_fetch_in_order(probe_server, tmp_path):
    endpoint, requests = probe_server
    assert proof.main(["--endpoint", endpoint, "--out-dir", str(tmp_path)]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == {"status": "PROOF_VALID", "reasons": []}
    assert summary["search_verified"] is True
    assert summary["discovery_verified"] is True
    assert summary["probe_content_hash_verified"] is True
    assert "x" * proof.PROBE_BYTE_COUNT not in (tmp_path / "summary.json").read_text(encoding="utf-8")

    paths = [urlsplit(path).path for _method, path, _body in requests]
    assert paths == [
        "/health",
        "/v1/namespaces",
        "/v1/search",
        "/v1/navigate/outline",
        "/v1/fetch",
    ]
    search_body = requests[2][2]
    fetch_body = requests[4][2]
    assert search_body == {
        "query": proof.PROBE_SEARCH_QUERY,
        "namespace": "skills",
        "limit": 10,
        "min_score": 0.0,
        "include_content": False,
        "maturity": "production",
    }
    assert fetch_body["by"] == "source_id"
    assert fetch_body["id"] == proof._PROBE_DOC_ID
    assert all(method in {"GET", "POST"} for method, _path, _body in requests)


def test_search_mismatch_stops_before_navigation_and_fetch(probe_server, tmp_path):
    endpoint, requests = probe_server
    original = _ProbeHandler.responses[("POST", "/v1/search")]
    _ProbeHandler.responses[("POST", "/v1/search")] = (
        original[0],
        {"results": [], "total": 0, "query": proof.PROBE_SEARCH_QUERY, "search_time_ms": 1.0},
        original[2],
    )
    try:
        assert proof.main(["--endpoint", endpoint, "--out-dir", str(tmp_path)]) == proof.CONTRACT_MISMATCH_CODE
    finally:
        _ProbeHandler.responses[("POST", "/v1/search")] = original
    assert [urlsplit(path).path for _method, path, _body in requests] == [
        "/health",
        "/v1/namespaces",
        "/v1/search",
    ]
    assert not (tmp_path / "summary.json").exists()


def test_non_loopback_endpoint_is_rejected_without_a_request(tmp_path):
    assert proof.main(
        ["--endpoint", "http://100.70.137.104:8310", "--out-dir", str(tmp_path)]
    ) == proof.INVALID_EVIDENCE_CODE


def test_http_error_body_is_not_disclosed(probe_server, tmp_path, capsys):
    endpoint, _requests = probe_server
    _ProbeHandler.responses[("GET", "/health")] = (
        500,
        "sensitive-response-body",
        "text/plain",
    )
    try:
        assert proof.main(["--endpoint", endpoint, "--out-dir", str(tmp_path)]) == proof.INVALID_EVIDENCE_CODE
    finally:
        _ProbeHandler.responses[("GET", "/health")] = (
            200,
            {"service": "Scrutator", "version": "test"},
            "application/json",
        )
    assert "sensitive-response-body" not in capsys.readouterr().err


def test_source_has_no_operator_escalation_or_raw_http_body_path():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "operator authorization" not in source.lower()
    assert "operator must" not in source.lower()
    assert "raw[:" not in source
    assert '"/v1/index"' not in source
    assert "http://127.0.0.1:" in source


MUTANTS = (
    (
        "search-document-id",
        'if result.get("source_id") != _PROBE_DOC_ID:',
        "if False:",
        "_check_probe_search",
        {"source_id": "wrong-id"},
    ),
    (
        "discovery-document-id",
        'if payload.get("doc_id") != _PROBE_DOC_ID:',
        "if False:",
        "_check_probe_discovery",
        {"doc_id": "wrong-id"},
    ),
    (
        "fetch-content-hash",
        "if actual_hash != PROBE_CONTENT_HASH:",
        "if False:",
        "_check_probe_fetch",
        {"content_hash": "sha256:" + "0" * 64},
    ),
)


def test_contract_mutants_are_killed(tmp_path):
    child = """
import importlib.util
import os

path = os.environ["MUTANT_MODULE"]
spec = importlib.util.spec_from_file_location("mutant", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fake_request(url, *, body=None, timeout_s=30.0):
    if url.endswith("/v1/search"):
        return ({"results": [{
            "source_path": module.PROBE_SOURCE_PATH,
            "source_id": os.environ["SEARCH_ID"],
            "content_hash": module.PROBE_CONTENT_HASH,
            "namespace": module.REQUIRED_NAMESPACE,
        }]}, 1.0)
    if "/v1/navigate/outline?" in url:
        return ({
            "source_path": module.PROBE_SOURCE_PATH,
            "namespace": module.REQUIRED_NAMESPACE,
            "doc_id": os.environ["DISCOVERY_ID"],
            "total_chunks": 1,
            "outline": [{}],
        }, 1.0)
    return ({
        "content_hash": os.environ["FETCH_HASH"],
        "content": "x" * module.PROBE_BYTE_COUNT,
        "namespace": module.REQUIRED_NAMESPACE,
        "trust_class": "skill",
        "content_exact": True,
        "path": module.PROBE_SOURCE_PATH,
    }, 1.0)

module._request_json = fake_request
function = os.environ["MUTANT_FUNCTION"]
try:
    getattr(module, function)("http://127.0.0.1:1", 1.0)
except module.ContractMismatch:
    raise SystemExit(0)
raise SystemExit(1)
"""

    for name, needle, replacement, function, wrong in MUTANTS:
        source = SCRIPT.read_text(encoding="utf-8")
        assert source.count(needle) == 1, name
        mutated = tmp_path / f"{name}.py"
        mutated.write_text(source.replace(needle, replacement), encoding="utf-8")
        env = {
            **os.environ,
            "MUTANT_MODULE": str(mutated),
            "MUTANT_FUNCTION": function,
            "SEARCH_ID": wrong.get("source_id", proof._PROBE_DOC_ID),
            "DISCOVERY_ID": wrong.get("doc_id", proof._PROBE_DOC_ID),
            "FETCH_HASH": wrong.get("content_hash", proof.PROBE_CONTENT_HASH),
        }
        result = subprocess.run(
            [sys.executable, "-c", child],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0, f"mutation survived: {name}\n{result.stderr}"
