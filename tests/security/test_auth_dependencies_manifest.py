"""SRCH-0023 V-AC-10 — Auth Arcana dependency manifest is present and well-formed."""

from __future__ import annotations

from pathlib import Path

import yaml


class TestAuthDependenciesManifest:
    def test_manifest_exists_at_repo_root(self):
        repo_root = Path(__file__).resolve().parents[2]
        manifest = repo_root / "auth.dependencies.yaml"
        assert manifest.exists(), "auth.dependencies.yaml must exist at repo root (mandate §6)"

    def test_manifest_is_valid_yaml_with_required_fields(self):
        repo_root = Path(__file__).resolve().parents[2]
        manifest = repo_root / "auth.dependencies.yaml"
        data = yaml.safe_load(manifest.read_text())
        assert data["service"] == "scrutator"
        assert data["auth_arcana_min_version"] == "0.5.0"
        assert data["last_validated"] == "2026-07-15"
        assert data["identity"]["oidc_issuer"] == "https://auth.arcanada.ai"
        assert data["identity"]["resource_audience"] == "urn:arcanada:scrutator:ltm"
        assert data["identity"]["m2m_client_id"] == "muneral-kb-sync"
        assert data["identity"]["m2m_algorithms"] == ["EdDSA"]
        assert "kb:ltm.read" in data["scopes_consumed"]
        assert data["grace_period"]["flag"] == "SCRUTATOR_AUTH_ENFORCE"
        assert data["grace_period"]["default"] is False
