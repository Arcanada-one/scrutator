"""Safety-contract tests for the SRCH-0031 loopback measurement wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import granted_context_app
import pytest


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        return None


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


@pytest.mark.asyncio
async def test_build_context_uses_live_grants_and_requires_arcanada(monkeypatch):
    connection = AsyncMock()
    connection.fetch.return_value = [{"id": 7, "name": "arcanada"}]
    monkeypatch.setenv("SCRUTATOR_BENCHMARK_PRINCIPAL", "kb-observer")
    monkeypatch.setattr(
        granted_context_app,
        "resolve_allowed_namespaces",
        AsyncMock(return_value=frozenset({7})),
    )
    monkeypatch.setattr(granted_context_app, "get_pool", AsyncMock(return_value=_Pool(connection)))

    context = await granted_context_app.build_benchmark_context()

    assert context.principal_id == "kb-observer"
    assert context.allowed_namespace_ids == frozenset({7})
    assert context.allowed_namespace_names == frozenset({"arcanada"})
    connection.fetch.assert_awaited_once()

    connection.fetch.return_value = [{"id": 8, "name": "other"}]
    with pytest.raises(RuntimeError, match="not granted"):
        await granted_context_app.build_benchmark_context()


def test_shell_runner_is_loopback_lifespan_off_and_fail_closed():
    script = (Path(__file__).resolve().parent.parent / "live" / "run_rerank_gate.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "trap cleanup EXIT" in script
    assert "--network host" in script
    assert "--host 127.0.0.1" in script
    assert "--lifespan off" in script
    assert '"production_image": production_image' in script
    assert "SCRUTATOR_BENCHMARK_SCOPE" in script
    assert "--eligibility-scope" in script
    assert 'off_id=""' in script
    assert 'on_id=""' in script
    assert 'docker rm -f "$on_container" "$off_container"' not in script
    assert 'docker rm -f "$on_id"' in script
    assert 'docker rm -f "$off_id"' in script
    assert "authorized" in script
    assert "chunk_count" not in script
    assert "summary.json" in script
    assert "flock -n" in script
    assert "/run/user/" in script
    assert "lock path must not be a symlink" in script
    assert "trap unexpected_failure ERR" in script
    assert "trap - ERR" in script
    assert "SCRUTATOR_RERANK_ENABLED=false" in script
    assert "SCRUTATOR_RERANK_ENABLED=true" in script
    assert "SCRUTATOR_DATABASE_POOL_MIN=1" in script
    assert "SCRUTATOR_DATABASE_POOL_MAX=2" in script
    for secret_name in (
        "SCRUTATOR_LTM_WRITER_TOKEN",
        "SCRUTATOR_FEEDER_TOKEN",
        "SCRUTATOR_ROLLBACK_TOKEN",
        "SCRUTATOR_OPERATOR_ROLLBACK_TOKEN",
    ):
        assert f"{secret_name}=" in script
    assert "0.0.0.0" not in script
    assert "StrictHostKeyChecking=no" not in script


def test_port_collision_does_not_remove_another_runs_containers(tmp_path):
    script = Path(__file__).resolve().parent.parent / "live" / "run_rerank_gate.sh"
    source = tmp_path / "source"
    (source / "live").mkdir(parents=True)
    (source / "rerank_gate.py").write_text("# probe\n", encoding="utf-8")
    (source / "golden-arcanada-v0.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "live" / "granted_context_app.py").write_text("# probe\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"

    docker = fake_bin / "docker"
    docker.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{docker_log}"\n', encoding="utf-8")
    docker.chmod(0o755)
    ss = fake_bin / "ss"
    ss.write_text('#!/bin/sh\nprintf "LISTEN 0 128 127.0.0.1:18310 0.0.0.0:*\\n"\n', encoding="utf-8")
    ss.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(script),
            f"scrutator-deploy:{'a' * 40}",
            str(source),
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 2
    assert "benchmark loopback port already in use" in completed.stderr
    assert not docker_log.exists()


def test_simultaneous_start_fails_closed_before_docker(tmp_path):
    script = Path(__file__).resolve().parent.parent / "live" / "run_rerank_gate.sh"
    source = tmp_path / "source"
    (source / "live").mkdir(parents=True)
    (source / "rerank_gate.py").write_text("# probe\n", encoding="utf-8")
    (source / "golden-arcanada-v0.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "live" / "granted_context_app.py").write_text("# probe\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"

    docker = fake_bin / "docker"
    docker.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{docker_log}"\n', encoding="utf-8")
    docker.chmod(0o755)
    flock = fake_bin / "flock"
    flock.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    flock.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(script),
            f"scrutator-deploy:{'a' * 40}",
            str(source),
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SCRUTATOR_BENCHMARK_LOCK_FILE": str(tmp_path / "gate.lock"),
        },
    )

    assert completed.returncode == 2
    assert "another SRCH-0031 benchmark owns the fixed ports" in completed.stderr
    assert not docker_log.exists()


def test_precreated_lock_symlink_is_rejected_without_truncating_target(tmp_path):
    script = Path(__file__).resolve().parent.parent / "live" / "run_rerank_gate.sh"
    source = tmp_path / "source"
    (source / "live").mkdir(parents=True)
    (source / "rerank_gate.py").write_text("# probe\n", encoding="utf-8")
    (source / "golden-arcanada-v0.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "live" / "granted_context_app.py").write_text("# probe\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("preserve-me\n", encoding="utf-8")
    lock_path = tmp_path / "gate.lock"
    lock_path.symlink_to(victim)

    completed = subprocess.run(
        [
            "bash",
            str(script),
            f"scrutator-deploy:{'a' * 40}",
            str(source),
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "SCRUTATOR_BENCHMARK_LOCK_FILE": str(lock_path)},
    )

    assert completed.returncode == 2
    assert "lock path must not be a symlink" in completed.stderr
    assert victim.read_text(encoding="utf-8") == "preserve-me\n"


def test_post_measurement_shell_failure_cannot_be_keep_off(tmp_path):
    script = Path(__file__).resolve().parent.parent / "live" / "run_rerank_gate.sh"
    image = f"scrutator-deploy:{'a' * 40}"
    source = tmp_path / "source"
    (source / "live").mkdir(parents=True)
    (source / "golden-arcanada-v0.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "live" / "granted_context_app.py").write_text("# probe\n", encoding="utf-8")
    (source / "rerank_gate.py").write_text(
        """
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[sys.argv.index("--out-dir") + 1])
(out_dir / "summary.json").write_text(
    json.dumps({"verdict": {"status": "KEEP_OFF", "reasons": ["probe"]}}),
    encoding="utf-8",
)
raise SystemExit(1)
""".lstrip(),
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    docker = fake_bin / "docker"
    docker.write_text(
        f"""#!/bin/bash
if [[ "$1" == "inspect" && "$2" == "--format" ]]; then
    if [[ "$3" == *Config.Image* ]]; then
        printf '%s\\n' '{image}'
    else
        printf '%s\\n' 'production-container-id'
    fi
elif [[ "$1" == "inspect" ]]; then
    printf '%s\\n' '[{{"Config":{{"Env":[]}}}}]'
elif [[ "$1" == "image" && "$2" == "inspect" ]]; then
    printf '%s\\n' 'sha256:candidate-image-id'
elif [[ "$1" == "run" ]]; then
    printf '%s\\n' "owned-$RANDOM"
elif [[ "$1" == "logs" ]]; then
    exit 1
elif [[ "$1" == "rm" ]]; then
    exit 0
fi
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    ss = fake_bin / "ss"
    ss.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    ss.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/bin/bash
url="${!#}"
if [[ "$url" == */v1/namespaces ]]; then
    printf '%s\\n' '[{"name":"arcanada"}]'
else
    printf '%s\\n' '{"status":"ok"}'
fi
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(script), image, str(source), str(tmp_path / "output")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SCRUTATOR_BENCHMARK_LOCK_FILE": str(tmp_path / "gate.lock"),
        },
    )

    assert completed.returncode == 2
    assert "unexpected shell failure; evidence is invalid" in completed.stderr
