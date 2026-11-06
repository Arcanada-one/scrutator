"""KC-PIPE-S3: contract + negative tests for verified-byte retrieval.

Hermetic: in-process FastAPI client, fixture LocationRegistry/v1 + temp files.
Does not claim live arcana-dbs library fetch (that stays NOT_MEASURED until S1/S2 freeze).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.kc_retrieval.router import reset_rate_limiter_for_tests
from tests.conftest import make_tenant_context, override_tenant_context


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_id(data: bytes) -> str:
    return f"d_{_sha(data)}"


@pytest.fixture
def client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def retrieval_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Keep happy-path bodies under the fixture size budget (32 bytes).
    original = b"kc-public-fixture-bytes\n"  # 24 bytes
    corrupt_expected = b"expected-but-file-was-tampered\n"
    private = b"private-not-public\n"  # 19 bytes
    package_body = b'{"k":"pkg-stub"}\n'  # 17 bytes

    originals = tmp_path / "originals"
    packages = tmp_path / "packages"
    originals.mkdir()
    packages.mkdir()

    public_path = originals / "public.bin"
    public_path.write_bytes(original)
    public_id = _source_id(original)

    corrupt_path = originals / "corrupt.bin"
    corrupt_path.write_bytes(b"tampered-bytes-not-matching-id\n")
    corrupt_id = _source_id(corrupt_expected)  # id claims other bytes

    private_path = originals / "private.bin"
    private_path.write_bytes(private)
    private_id = _source_id(private)

    oversized = b"x" * 64
    oversized_path = originals / "oversized.bin"
    oversized_path.write_bytes(oversized)
    oversized_id = _source_id(oversized)

    package_path = packages / "pkg.json"
    package_path.write_bytes(package_body)
    package_id = f"{public_id}/book-package-stub"
    package_sha = _sha(package_body)

    mirror_path = originals / "public-mirror.bin"
    mirror_path.write_bytes(original)

    registry = {
        "schema": "LocationRegistry/v1",
        "contour": "devs-fixture:/tmp/source-library",
        "entries": [
            {
                "schema": "LocationRegistryRow/v1",
                "object_kind": "original",
                "object_id": public_id,
                "holder_host": "devs-fixture",
                "volume": str(tmp_path),
                "path": str(public_path),
                "role": "primary",
                "size_bytes": len(original),
                "verified_at": "2026-11-06T17:00:00Z",
                "verified_sha256": _sha(original),
            },
            {
                "schema": "LocationRegistryRow/v1",
                "object_kind": "original",
                "object_id": public_id,
                "holder_host": "devs-fixture",
                "volume": str(tmp_path),
                "path": str(mirror_path),
                "role": "mirror",
                "size_bytes": len(original),
                "verified_at": "2026-11-06T17:00:00Z",
                "verified_sha256": _sha(original),
            },
            {
                "schema": "LocationRegistryRow/v1",
                "object_kind": "original",
                "object_id": corrupt_id,
                "holder_host": "devs-fixture",
                "volume": str(tmp_path),
                "path": str(corrupt_path),
                "role": "primary",
                "size_bytes": corrupt_path.stat().st_size,
                "verified_at": "2026-11-06T17:00:00Z",
                "verified_sha256": _sha(corrupt_expected),
            },
            {
                "schema": "LocationRegistryRow/v1",
                "object_kind": "original",
                "object_id": private_id,
                "holder_host": "devs-fixture",
                "volume": str(tmp_path),
                "path": str(private_path),
                "role": "primary",
                "size_bytes": len(private),
                "verified_at": "2026-11-06T17:00:00Z",
                "verified_sha256": _sha(private),
            },
            {
                "schema": "LocationRegistryRow/v1",
                "object_kind": "original",
                "object_id": oversized_id,
                "holder_host": "devs-fixture",
                "volume": str(tmp_path),
                "path": str(oversized_path),
                "role": "primary",
                "size_bytes": len(oversized),
                "verified_at": "2026-11-06T17:00:00Z",
                "verified_sha256": _sha(oversized),
            },
            {
                "schema": "LocationRegistryRow/v1",
                "object_kind": "package",
                "object_id": package_id,
                "holder_host": "devs-fixture",
                "volume": str(tmp_path),
                "path": str(package_path),
                "role": "primary",
                "size_bytes": len(package_body),
                "verified_at": "2026-11-06T17:00:00Z",
                "verified_sha256": package_sha,
            },
        ],
    }
    registry_path = tmp_path / "location-registry" / "registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(settings, "kc_retrieval_enabled", True)
    monkeypatch.setattr(settings, "kc_retrieval_registry_path", str(registry_path))
    monkeypatch.setattr(settings, "kc_retrieval_namespace", "knowledge-contract")
    monkeypatch.setattr(settings, "kc_retrieval_public_object_ids", public_id)
    monkeypatch.setattr(settings, "kc_retrieval_max_bytes", 32)
    monkeypatch.setattr(settings, "kc_retrieval_rate_limit", 100)
    monkeypatch.setattr(settings, "kc_retrieval_rate_window_seconds", 60.0)
    reset_rate_limiter_for_tests(limit=100, window_seconds=60.0)

    return {
        "public_id": public_id,
        "public_bytes": original,
        "corrupt_id": corrupt_id,
        "private_id": private_id,
        "oversized_id": oversized_id,
        "package_id": package_id,
        "package_bytes": package_body,
        "package_sha": package_sha,
        "registry_path": registry_path,
        "public_path": public_path,
    }


def test_contract_retrieve_original_by_source_id(client, retrieval_fixture):
    from scrutator.health import app

    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        resp = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['public_id']}&view=original")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object_id"] == retrieval_fixture["public_id"]
    assert body["object_kind"] == "original"
    assert body["role_served"] == "primary"
    assert body["sha256"] == _sha(retrieval_fixture["public_bytes"])
    assert body["content"] == retrieval_fixture["public_bytes"].decode("utf-8")
    assert body["citation"]["object_id"] == retrieval_fixture["public_id"]
    assert body["citation"]["byte_offset_start"] == 0
    assert body["citation"]["byte_offset_end"] == len(retrieval_fixture["public_bytes"])
    assert body["kc2_admission"] == "not_measured"
    assert "path" not in body
    assert "path" not in body["citation"]


def test_contract_retrieve_package_and_fragment(client, retrieval_fixture):
    from scrutator.health import app

    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        pkg = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['package_id']}&view=package")
        frag = client.get(
            f"/v1/kc/retrieve?id={retrieval_fixture['public_id']}&view=fragment&offset_start=0&offset_end=9"
        )
    assert pkg.status_code == 200, pkg.text
    assert pkg.json()["view"] == "package"
    assert pkg.json()["sha256"] == retrieval_fixture["package_sha"]
    assert frag.status_code == 200, frag.text
    assert frag.json()["view"] == "fragment"
    assert frag.json()["content"] == retrieval_fixture["public_bytes"][:9].decode("utf-8")
    assert frag.json()["citation"]["byte_offset_end"] == 9


def test_contract_falls_back_to_mirror_when_primary_corrupt(client, retrieval_fixture, tmp_path):
    from scrutator.health import app

    # Corrupt the primary file in place; mirror remains valid.
    retrieval_fixture["public_path"].write_bytes(b"broken-primary\n")
    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        resp = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['public_id']}&view=original")
    assert resp.status_code == 200, resp.text
    assert resp.json()["role_served"] == "mirror"
    assert resp.json()["sha256"] == _sha(retrieval_fixture["public_bytes"])


def test_negative_unknown_id(client, retrieval_fixture):
    from scrutator.health import app

    unknown = "d_" + ("ab" * 32)
    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        resp = client.get(f"/v1/kc/retrieve?id={unknown}&view=original")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "not_found"


def test_negative_corrupt_copy(client, retrieval_fixture):
    from scrutator.health import app

    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        resp = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['corrupt_id']}&view=original")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "corrupt_copy"


def test_negative_private_item_without_grant(client, retrieval_fixture):
    from scrutator.health import app

    # Public-surface principal: no knowledge-contract namespace; private id not allowlisted.
    ctx = make_tenant_context(
        namespace_ids=frozenset(),
        namespace_names=frozenset(),
        principal_id="anonymous",
    )
    with override_tenant_context(app, ctx):
        private = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['private_id']}&view=original")
        public = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['public_id']}&view=original")
    assert private.status_code == 404
    assert private.json()["detail"] == "not_found"
    assert public.status_code == 200


def test_negative_oversized_request(client, retrieval_fixture):
    from scrutator.health import app

    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        resp = client.get(f"/v1/kc/retrieve?id={retrieval_fixture['oversized_id']}&view=original")
    assert resp.status_code == 413
    assert resp.json()["detail"] == "object exceeds size budget"


def test_no_write_route_on_kc_retrieval_prefix(client, retrieval_fixture):
    from scrutator.health import app

    ctx = make_tenant_context(
        namespace_ids=frozenset({42}),
        namespace_names=frozenset({"knowledge-contract"}),
    )
    with override_tenant_context(app, ctx):
        resp = client.post("/v1/kc/retrieve", json={"id": retrieval_fixture["public_id"]})
    assert resp.status_code in {404, 405}


def test_settings_default_off():
    from scrutator.config import Settings

    assert Settings(_env_file=None).kc_retrieval_enabled is False
    assert Settings(_env_file=None).kc_retrieval_namespace == "knowledge-contract"
