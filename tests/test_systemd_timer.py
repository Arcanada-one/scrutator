"""LTM-0025 Task 10: hardened isolated Muneral sync timer and immutable installer."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy"
SERVICE = DEPLOY / "muneral-kb-sync.service"
TIMER = DEPLOY / "muneral-kb-sync.timer"
RUNNER = DEPLOY / "muneral-kb-sync-run.sh"
INSTALLER = DEPLOY / "install-muneral-kb-sync.sh"
RUNTIME_REQUIREMENTS = DEPLOY / "requirements-muneral-kb-sync.txt"
PILOT_TASK_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000004"


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    shutil.copytree(REPO / "src" / "scrutator", source / "src" / "scrutator")
    shutil.copytree(REPO / "tools" / "muneral_sync", source / "tools" / "muneral_sync")
    (source / "tools" / "__init__.py").write_text("")
    (source / "deploy").mkdir()
    for artifact in (SERVICE, TIMER, RUNNER):
        shutil.copy2(artifact, source / "deploy" / artifact.name)
    shutil.copy2(RUNTIME_REQUIREMENTS, source / "deploy" / RUNTIME_REQUIREMENTS.name)
    _run(["git", "init", "-q", "-b", "main"], cwd=source)
    _run(["git", "config", "user.name", "Timer Test"], cwd=source)
    _run(["git", "config", "user.email", "timer-test@example.invalid"], cwd=source)
    _run(["git", "add", "."], cwd=source)
    committed = _run(["git", "commit", "-q", "-m", "fixture"], cwd=source)
    assert committed.returncode == 0, committed.stdout + committed.stderr
    return source


def _head(source: Path) -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=source)
    assert result.returncode == 0
    return result.stdout.strip()


def _next_commit(source: Path, marker: str) -> str:
    marker_file = source / "src" / "scrutator" / "release_marker.py"
    marker_file.write_text(f'MARKER = "{marker}"\n')
    _run(["git", "add", str(marker_file)], cwd=source)
    result = _run(["git", "commit", "-q", "-m", f"fixture {marker}"], cwd=source)
    assert result.returncode == 0, result.stdout + result.stderr
    return _head(source)


def test_service_has_exact_identity_credentials_and_incremental_entrypoint():
    body = SERVICE.read_text()
    assert "User=muneral-kb-sync" in body
    assert "Group=muneral-kb-sync" in body
    assert body.count("LoadCredential=") == 2
    assert "LoadCredential=muneral-db-dsn:/etc/muneral-kb-sync/muneral-db-dsn" in body
    assert "LoadCredential=ltm-writer-token:/etc/muneral-kb-sync/ltm-writer-token" in body
    assert "StateDirectory=muneral-kb-sync/runtime" in body
    assert "ExecStart=/opt/muneral-kb-sync/current/bin/muneral-kb-sync --incremental --timer" in body


def test_service_hardening_and_retry_are_bounded():
    body = SERVICE.read_text()
    for directive in (
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "NoNewPrivileges=yes",
        "ReadWritePaths=/var/lib/muneral-kb-sync/runtime",
        "TimeoutStartSec=10min",
        "Restart=on-failure",
        "RestartSec=30s",
        "StartLimitBurst=3",
    ):
        assert directive in body


def test_timer_is_bounded_persistent_and_not_self_enabling():
    body = TIMER.read_text()
    assert "Unit=muneral-kb-sync.service" in body
    assert "OnUnitInactiveSec=5min" in body
    assert "RandomizedDelaySec=30s" in body
    assert "Persistent=true" in body
    assert "WantedBy=timers.target" in body


def test_units_pass_systemd_analyze_verify_when_available(tmp_path):
    analyzer = shutil.which("systemd-analyze")
    if analyzer is None:
        pytest.skip("systemd-analyze is not installed")
    root = tmp_path / "root"
    unit_dir = root / "etc/systemd/system"
    executable = root / "opt/muneral-kb-sync/current/bin/muneral-kb-sync"
    unit_dir.mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    shutil.copytree("/usr/lib/systemd/system", root / "usr/lib/systemd/system", symlinks=True)
    shutil.copy2(SERVICE, unit_dir / SERVICE.name)
    shutil.copy2(TIMER, unit_dir / TIMER.name)
    output = _run(
        [
            analyzer,
            f"--root={root}",
            "verify",
            str(unit_dir / SERVICE.name),
            str(unit_dir / TIMER.name),
        ]
    )
    assert output.returncode == 0, output.stdout + output.stderr


def _runner_fixture(tmp_path: Path) -> tuple[dict[str, str], Path]:
    credentials = tmp_path / "credentials"
    state = tmp_path / "state"
    credentials.mkdir()
    state.mkdir()
    (credentials / "muneral-db-dsn").write_text("dsn-secret-sentinel\n")
    (credentials / "ltm-writer-token").write_text("writer-secret-sentinel\n")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$@" >"$RUNNER_ARGS_FILE"\n'
        '[ -n "${RUNNER_BLOCK_FILE:-}" ] && {\n'
        'touch "$RUNNER_BLOCK_FILE.ready"\n'
        'while [ -e "$RUNNER_BLOCK_FILE" ]; do sleep 0.05; done\n'
        "}\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    args_file = tmp_path / "args"
    env = {
        **os.environ,
        "CREDENTIALS_DIRECTORY": str(credentials),
        "MUNERAL_KB_SYNC_STATE_DIR": str(state),
        "MUNERAL_KB_SYNC_PYTHON": str(fake_python),
        "RUNNER_ARGS_FILE": str(args_file),
    }
    return env, args_file


def test_runner_passes_only_credential_paths_and_preserves_full_mode_args(tmp_path):
    env, args_file = _runner_fixture(tmp_path)
    output = _run([str(RUNNER), "--all", "--operator-go", "FULL-MUNERAL-BACKFILL"], env=env)
    combined = output.stdout + output.stderr + args_file.read_text()
    assert output.returncode == 0
    assert "--all" in args_file.read_text()
    assert "FULL-MUNERAL-BACKFILL" in args_file.read_text()
    assert "--dsn-credential" in args_file.read_text()
    assert "--writer-credential" in args_file.read_text()
    assert "dsn-secret-sentinel" not in combined
    assert "writer-secret-sentinel" not in combined


def test_release_runtime_requirements_are_fully_pinned():
    blocks = [block for block in RUNTIME_REQUIREMENTS.read_text().split("\n\n") if not block.startswith("#")]
    assert blocks
    assert all("==" in block and "--hash=sha256:" in block for block in blocks)
    installer = INSTALLER.read_text()
    assert "--require-hashes" in installer
    assert "--only-binary=:all:" in installer


def test_runner_defaults_to_the_release_internal_python():
    body = RUNNER.read_text()
    assert "MUNERAL_KB_SYNC_PYTHON:-${release_root}/venv/bin/python" in body
    assert "MUNERAL_KB_SYNC_STATE_DIR:-/var/lib/muneral-kb-sync/runtime" in body


def test_runner_default_endpoint_is_the_colocated_loopback_service(tmp_path):
    env, args_file = _runner_fixture(tmp_path)
    env.pop("MUNERAL_KB_SYNC_ENDPOINT", None)
    output = _run([str(RUNNER), "--incremental"], env=env)
    assert output.returncode == 0, output.stdout + output.stderr
    arguments = args_file.read_text().splitlines()
    endpoint = arguments[arguments.index("--endpoint") + 1]
    parsed = urlsplit(endpoint)
    assert parsed.hostname == "127.0.0.1"
    assert ip_address(parsed.hostname).is_loopback
    assert endpoint == "http://127.0.0.1:8310/v1/ltm/ingest"


@pytest.mark.parametrize("flag", ["--dsn-credential", "--writer-credential", "--cursor-file", "--endpoint"])
def test_runner_rejects_overrides_of_wrapper_controlled_paths(tmp_path, flag):
    env, _ = _runner_fixture(tmp_path)
    output = _run([str(RUNNER), "--incremental", flag, "/tmp/untrusted"], env=env)
    assert output.returncode == 64


def test_runner_uses_one_nonblocking_lock_for_all_modes(tmp_path):
    env, _ = _runner_fixture(tmp_path)
    block = tmp_path / "block"
    block.touch()
    env["RUNNER_BLOCK_FILE"] = str(block)
    first = subprocess.Popen(
        [str(RUNNER), "--incremental"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready = Path(f"{block}.ready")
    for _ in range(100):
        if ready.exists():
            break
        import time

        time.sleep(0.02)
    assert ready.exists()
    second = _run([str(RUNNER), "--all", "--dry-run"], env=env)
    block.unlink()
    first_stdout, first_stderr = first.communicate(timeout=5)
    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 75


def _install(root: Path, source: Path, sha: str | None = None, *extra: str, env=None):
    sha = sha or _head(source)
    return _run(
        [str(INSTALLER), "install", "--root", str(root), "--sha", sha, "--source", str(source), *extra],
        env=env,
    )


def test_installer_rejects_symlinked_or_writable_canonical_directory(tmp_path, source_repo):
    root = tmp_path / "root"
    target = tmp_path / "redirect"
    target.mkdir()
    (root / "opt").mkdir(parents=True)
    (root / "opt/muneral-kb-sync").symlink_to(target, target_is_directory=True)
    assert _install(root, source_repo).returncode != 0

    (root / "opt/muneral-kb-sync").unlink()
    canonical = root / "opt/muneral-kb-sync"
    canonical.mkdir()
    canonical.chmod(0o777)
    assert _install(root, source_repo).returncode != 0


def test_installer_account_validation_is_fail_closed():
    body = INSTALLER.read_text()
    for contract in ("uid -ne 0", "gid -ne 0", "/usr/sbin/nologin", "supplementary", "primary_gid"):
        assert contract in body


def test_installer_global_lock_rejects_concurrent_transaction(tmp_path, source_repo):
    root = tmp_path / "root"
    hold = tmp_path / "hold"
    hold.touch()
    env = {**os.environ, "MUNERAL_KB_SYNC_TEST_HOLD_LOCK_FILE": str(hold)}
    first = subprocess.Popen(
        [str(INSTALLER), "rollback", "--root", str(root)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready = Path(f"{hold}.ready")
    for _ in range(100):
        if ready.exists():
            break
        import time

        time.sleep(0.02)
    assert ready.exists()
    second = _install(root, source_repo)
    assert second.returncode == 75
    hold.unlink()
    first.communicate(timeout=5)


def test_installer_creates_self_contained_release_and_atomic_links(tmp_path, source_repo):
    root = tmp_path / "root"
    first_sha = _head(source_repo)
    first = _install(root, source_repo)
    assert first.returncode == 0, first.stdout + first.stderr
    opt = root / "opt/muneral-kb-sync"
    first_release = opt / "releases" / first_sha
    assert (opt / "current").resolve() == first_release.resolve()
    assert not (root / "var/lib/muneral-kb-sync/runtime/timer-enabled").exists()
    assert stat.S_IMODE((first_release / "bin/muneral-kb-sync").stat().st_mode) == 0o555
    assert stat.S_IMODE(first_release.stat().st_mode) == 0o555
    assert (first_release / "RELEASE_SHA").read_text() == f"{first_sha}\n"
    assert (first_release / "PAYLOAD_SHA256SUMS").is_file()
    smoke = _run(
        [
            str(first_release / "venv/bin/python"),
            "-c",
            "import asyncpg,httpx,tools.muneral_sync.cli",
        ],
        env={**os.environ, "PYTHONPATH": f"{first_release / 'src'}:{first_release}"},
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr

    second_sha = _next_commit(source_repo, "second")
    second = _install(root, source_repo, second_sha)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (opt / "current").resolve() == (opt / "releases" / second_sha).resolve()
    assert (opt / "previous").resolve() == first_release.resolve()


def test_installer_rejects_fabricated_sha_and_dirty_tracked_source(tmp_path, source_repo):
    root = tmp_path / "root"
    fabricated = _install(root, source_repo, "f" * 40)
    assert fabricated.returncode != 0
    tracked = source_repo / "deploy" / RUNNER.name
    tracked.write_text(tracked.read_text() + "# dirty\n")
    dirty = _install(root, source_repo, _head(source_repo))
    assert dirty.returncode != 0
    assert not (root / "opt/muneral-kb-sync/current").exists()


def test_installer_rejects_tampered_existing_release(tmp_path, source_repo):
    root = tmp_path / "root"
    sha = _head(source_repo)
    assert _install(root, source_repo, sha).returncode == 0
    release = root / "opt/muneral-kb-sync/releases" / sha
    target = release / "tools/muneral_sync/cli.py"
    target.chmod(0o644)
    target.write_text(target.read_text() + "# tampered\n")
    rejected = _install(root, source_repo, sha)
    assert rejected.returncode != 0


@pytest.mark.parametrize("tamper", ["directory-mode", "executable-mode", "special-node"])
def test_installer_rejects_release_metadata_or_node_tamper(tmp_path, source_repo, tamper):
    root = tmp_path / "root"
    sha = _head(source_repo)
    assert _install(root, source_repo, sha).returncode == 0
    release = root / "opt/muneral-kb-sync/releases" / sha
    if tamper == "directory-mode":
        release.chmod(0o575)
    elif tamper == "executable-mode":
        (release / "bin/muneral-kb-sync").chmod(0o455)
    else:
        release.chmod(0o755)
        os.mkfifo(release / "unexpected-fifo")
        release.chmod(0o555)
    assert _install(root, source_repo, sha).returncode != 0


def test_rollback_rejects_tampered_previous_release(tmp_path, source_repo):
    root = tmp_path / "root"
    first_sha = _head(source_repo)
    assert _install(root, source_repo, first_sha).returncode == 0
    second_sha = _next_commit(source_repo, "tamper-rollback")
    assert _install(root, source_repo, second_sha).returncode == 0
    previous_file = root / "opt/muneral-kb-sync/releases" / first_sha / "tools/muneral_sync/cli.py"
    previous_file.chmod(0o644)
    previous_file.write_text(previous_file.read_text() + "# tampered\n")
    rejected = _run([str(INSTALLER), "rollback", "--root", str(root)])
    assert rejected.returncode != 0
    current = root / "opt/muneral-kb-sync/current"
    assert current.resolve() == (root / "opt/muneral-kb-sync/releases" / second_sha).resolve()


def test_installer_rollback_atomically_swaps_current_and_previous(tmp_path, source_repo):
    root = tmp_path / "root"
    first_sha = _head(source_repo)
    assert _install(root, source_repo).returncode == 0
    second_sha = _next_commit(source_repo, "rollback")
    assert _install(root, source_repo, second_sha).returncode == 0
    result = _run([str(INSTALLER), "rollback", "--root", str(root)])
    assert result.returncode == 0, result.stdout + result.stderr
    opt = root / "opt/muneral-kb-sync"
    assert (opt / "current").resolve() == (opt / "releases" / first_sha).resolve()
    assert (opt / "previous").resolve() == (opt / "releases" / second_sha).resolve()


def _write_pilot_proof(path: Path, release_sha: str, *, lines: list[str] | None = None) -> None:
    canonical = lines or [
        f"task_id={PILOT_TASK_ID}",
        f"release_sha={release_sha}",
        "principal=muneral-kb-sync",
        "graph_proven=true",
        "recall_proven=true",
        "idempotent=true",
    ]
    path.write_text("\n".join(canonical) + "\n")
    path.chmod(0o600)


def test_installer_requires_exact_root_controlled_pilot_proof_before_timer_enable(tmp_path, source_repo):
    root = tmp_path / "root"
    sha = _head(source_repo)
    denied = _install(root, source_repo, sha, "--enable-timer-after-pilot")
    assert denied.returncode != 0
    state = root / "var/lib/muneral-kb-sync/runtime"
    state.mkdir(parents=True, exist_ok=True)
    _write_pilot_proof(state / "pilot-proven", sha)
    still_denied = _install(root, source_repo, sha, "--enable-timer-after-pilot")
    assert still_denied.returncode != 0
    credentials = root / "etc/muneral-kb-sync"
    for name in ("muneral-db-dsn", "ltm-writer-token"):
        credential = credentials / name
        credential.write_text("test-only-secret\n")
        credential.chmod(0o600)
    forged_with_credentials = _install(root, source_repo, sha, "--enable-timer-after-pilot")
    assert forged_with_credentials.returncode != 0
    (state / "pilot-proven").unlink()
    proof = credentials / "pilot-proven"
    _write_pilot_proof(proof, sha)
    allowed = _install(root, source_repo, sha, "--enable-timer-after-pilot")
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert (state / "timer-enabled").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda lines: lines[:-1],
        lambda lines: [*lines, "extra=true"],
        lambda lines: [*lines, lines[-1]],
        lambda lines: [*lines[:3], "graph_proven=false", *lines[4:]],
        lambda lines: [*lines[:1], "release_sha=" + "0" * 40, *lines[2:]],
        lambda lines: [lines[1], lines[0], *lines[2:]],
    ],
)
def test_installer_rejects_malformed_or_unbound_pilot_proof(tmp_path, source_repo, mutate):
    root = tmp_path / "root"
    sha = _head(source_repo)
    assert _install(root, source_repo, sha).returncode == 0
    credentials = root / "etc/muneral-kb-sync"
    for name in ("muneral-db-dsn", "ltm-writer-token"):
        credential = credentials / name
        credential.write_text("test-only-secret\n")
        credential.chmod(0o600)
    canonical = [
        f"task_id={PILOT_TASK_ID}",
        f"release_sha={sha}",
        "principal=muneral-kb-sync",
        "graph_proven=true",
        "recall_proven=true",
        "idempotent=true",
    ]
    _write_pilot_proof(credentials / "pilot-proven", sha, lines=mutate(canonical))
    denied = _install(root, source_repo, sha, "--enable-timer-after-pilot")
    assert denied.returncode != 0
    assert not (root / "var/lib/muneral-kb-sync/runtime/timer-enabled").exists()


def test_installer_rejects_pilot_proof_with_non_private_mode(tmp_path, source_repo):
    root = tmp_path / "root"
    sha = _head(source_repo)
    assert _install(root, source_repo, sha).returncode == 0
    credentials = root / "etc/muneral-kb-sync"
    for name in ("muneral-db-dsn", "ltm-writer-token"):
        credential = credentials / name
        credential.write_text("test-only-secret\n")
        credential.chmod(0o600)
    proof = credentials / "pilot-proven"
    _write_pilot_proof(proof, sha)
    proof.chmod(0o644)
    denied = _install(root, source_repo, sha, "--enable-timer-after-pilot")
    assert denied.returncode != 0


def test_installer_removes_partial_staging_on_failure(tmp_path, source_repo):
    root = tmp_path / "root"
    env = {**os.environ, "MUNERAL_KB_SYNC_TEST_FAIL_AFTER_COPY": "1"}
    sha = _head(source_repo)
    failed = _install(root, source_repo, sha, env=env)
    assert failed.returncode != 0
    release_dir = root / "opt/muneral-kb-sync/releases"
    assert not list(release_dir.glob(".*.tmp.*"))
    assert not (release_dir / sha).exists()


def test_installer_restores_links_and_removes_new_release_on_activation_failure(tmp_path, source_repo):
    root = tmp_path / "root"
    first_sha = _head(source_repo)
    assert _install(root, source_repo, first_sha).returncode == 0
    failed_sha = _next_commit(source_repo, "failed")
    env = {**os.environ, "MUNERAL_KB_SYNC_TEST_FAIL_AFTER_SWITCH": "1"}
    failed = _install(root, source_repo, failed_sha, env=env)
    assert failed.returncode != 0
    opt = root / "opt/muneral-kb-sync"
    assert (opt / "current").resolve() == (opt / "releases" / first_sha).resolve()
    assert not (opt / "previous").exists()
    assert not (opt / "releases" / failed_sha).exists()


def test_installer_restores_units_and_timer_state_after_enable_failure(tmp_path, source_repo):
    root = tmp_path / "root"
    first_sha = _head(source_repo)
    assert _install(root, source_repo, first_sha).returncode == 0
    unit_dir = root / "etc/systemd/system"
    legacy_service = b"legacy service bytes\n"
    legacy_timer = b"legacy timer bytes\n"
    (unit_dir / SERVICE.name).write_bytes(legacy_service)
    (unit_dir / TIMER.name).write_bytes(legacy_timer)
    state = root / "var/lib/muneral-kb-sync/runtime"
    (state / "timer-enabled").touch()
    (state / "timer-active").touch()

    failed_sha = _next_commit(source_repo, "enable-failure")
    credentials = root / "etc/muneral-kb-sync"
    for name in ("muneral-db-dsn", "ltm-writer-token"):
        credential = credentials / name
        credential.write_text("test-only-secret\n")
        credential.chmod(0o600)
    _write_pilot_proof(credentials / "pilot-proven", failed_sha)
    env = {**os.environ, "MUNERAL_KB_SYNC_TEST_FAIL_AFTER_ENABLE": "1"}
    failed = _install(root, source_repo, failed_sha, "--enable-timer-after-pilot", env=env)
    assert failed.returncode != 0
    assert (unit_dir / SERVICE.name).read_bytes() == legacy_service
    assert (unit_dir / TIMER.name).read_bytes() == legacy_timer
    assert (state / "timer-enabled").exists()
    assert (state / "timer-active").exists()
