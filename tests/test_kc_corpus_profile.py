"""KB-CONS-02: the kc-corpus instance renders its own isolated profile (AC1).

The render comes from `docker compose config --format json` when the docker CLI with the compose
plugin is present (client-side, no daemon contacted); otherwise the file is parsed with PyYAML. The root
docker-compose.yml is rendered too, to prove its broad feeder list and fixed healthcheck are not inherited.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "instances" / "kc-corpus" / "docker-compose.yml"
FIXTURE_ENV = (
    "SCRUTATOR_DATABASE_URL=postgresql://kc_corpus_scrutator_app:fixture@100.97.136.74:5432/kc_corpus\n"
    "SCRUTATOR_FEEDER_TOKEN=fixture-feeder-token-not-a-secret\n"
)
BUILD_SHA = "0123456789abcdef0123456789abcdef01234567"


def _render() -> dict:
    docker = shutil.which("docker")
    if docker:
        with tempfile.TemporaryDirectory() as tmp:
            # Keep the relative build context valid: render from a copy at the same depth.
            base = Path(tmp)
            target = base / "instances" / "kc-corpus"
            target.mkdir(parents=True)
            shutil.copy(COMPOSE, target / "docker-compose.yml")
            (target / ".env").write_text(FIXTURE_ENV)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": tmp,
                "BUILD_SHA": BUILD_SHA,
                "DOCKER_HOST": "unix:///nonexistent",
            }
            try:
                out = subprocess.run(
                    [docker, "compose", "-f", str(target / "docker-compose.yml"), "config", "--format", "json"],
                    cwd=target,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                out = None
            if out is not None and out.returncode == 0:
                return json.loads(out.stdout)
    raw = yaml.safe_load(COMPOSE.read_text())
    svc = raw["services"]["scrutator"]
    env = {k: v for k, v in svc["environment"].items()}
    for key, value in list(env.items()):
        if isinstance(value, str) and value.startswith("${"):
            env[key] = "<interpolated>"
    return {"name": raw["name"], "services": {"scrutator": {**svc, "environment": env}}}


@pytest.fixture(scope="module")
def service() -> dict:
    return _render()["services"]["scrutator"]


def test_project_and_container_identity_are_dedicated(service):
    rendered = _render()
    assert rendered["name"] == "kc-corpus-scrutator"
    assert service["container_name"] == "kc-corpus-scrutator-scrutator-1"
    assert service.get("network_mode") in (None, "")  # never host network
    assert not service.get("volumes")


def test_port_is_loopback_only_and_distinct_from_shared_and_kc(service):
    ports = service["ports"]
    assert len(ports) == 1
    port = ports[0]
    if isinstance(port, dict):
        assert (port["host_ip"], int(port["published"]), int(port["target"])) == ("127.0.0.1", 8312, 8310)
    else:
        assert port == "127.0.0.1:8312:8310"
    assert 8312 not in (8310, 8311)


def test_feeder_scope_is_only_kc_corpus(service):
    env = service["environment"]
    assert env["SCRUTATOR_FEEDER_NAMESPACES"] == "kc-corpus"
    root_compose = (ROOT / "docker-compose.yml").read_text()
    # The root file appends many shared namespaces; none of them may appear in this profile.
    root_line = root_compose.split("SCRUTATOR_FEEDER_NAMESPACES:")[1].split("\n")[0]
    for name in re.findall(r",([a-z0-9-]+)(?=[,\"])", root_line):
        assert name not in COMPOSE.read_text().replace("kc-corpus", ""), name
    assert "${SCRUTATOR_FEEDER_NAMESPACES" not in COMPOSE.read_text()


def test_exact_evidence_pair_and_required_namespace(service):
    env = service["environment"]
    assert env["SCRUTATOR_EVIDENCE_EXACT_BYTES"] == "true"
    assert json.loads(env["SCRUTATOR_EVIDENCE_EXACT_NAMESPACES"]) == ["kc-corpus"]
    assert json.loads(env["SCRUTATOR_EVIDENCE_EXACT_REQUIRED_NAMESPACES"]) == ["kc-corpus"]


def test_auth_is_enforced_and_api_key_introspection_is_absent(service):
    env = service["environment"]
    assert env["SCRUTATOR_AUTH_ENFORCE"] == "true"
    for key in (
        "SCRUTATOR_AUTH_ARCANA_INTROSPECT_URL",
        "SCRUTATOR_AUTH_SERVICE_AUDIENCE",
        "SCRUTATOR_AUTH_SERVICE_SCOPE",
        "SCRUTATOR_PROJECT_WRITE_NAMESPACE",
        "SCRUTATOR_LTM_WRITER_TOKEN",
        "SCRUTATOR_ROLLBACK_TOKEN",
        "SCRUTATOR_OPERATOR_ROLLBACK_TOKEN",
        "SCRUTATOR_CAPABILITY_PROJECTION_TOKEN",
    ):
        assert key not in env, key


def test_healthcheck_targets_its_own_container_port_not_the_shared_listener(service):
    test = service["healthcheck"]["test"]
    assert test[0] == "CMD"
    assert "http://127.0.0.1:8310/health" in test  # container-internal port; the published port is 8312
    assert all("8311" not in part for part in test)


def test_database_and_feeder_capability_are_required_from_the_protected_env(service):
    text = COMPOSE.read_text()
    assert "${SCRUTATOR_DATABASE_URL:?" in text
    assert "${SCRUTATOR_FEEDER_TOKEN:?" in text
    assert "fixture" not in text


def test_image_revision_comes_from_broker_build_sha(service):
    text = COMPOSE.read_text()
    assert text.count("${BUILD_SHA:?") >= 2
    assert "org.opencontainers.image.revision" in text


def test_root_compose_is_untouched_by_the_profile():
    root = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svc = root["services"]["scrutator"]
    assert svc["network_mode"] == "host"
    assert "kc-corpus" not in (ROOT / "docker-compose.yml").read_text()
    assert svc["healthcheck"]["test"][-1] == "http://localhost:8310/health"
