"""LTM-0025 Task 10: hardened isolated Muneral sync timer and immutable installer."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy"
SERVICE = DEPLOY / "muneral-kb-sync.service"
TIMER = DEPLOY / "muneral-kb-sync.timer"
RUNNER = DEPLOY / "muneral-kb-sync-run.sh"
INSTALLER = DEPLOY / "install-muneral-kb-sync.sh"


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)


def test_service_has_exact_identity_credentials_and_incremental_entrypoint():
    body = SERVICE.read_text()
    assert "User=muneral-kb-sync" in body
    assert "Group=muneral-kb-sync" in body
    assert body.count("LoadCredential=") == 2
    assert "LoadCredential=muneral-db-dsn:/etc/muneral-kb-sync/muneral-db-dsn" in body
    assert "LoadCredential=ltm-writer-token:/etc/muneral-kb-sync/ltm-writer-token" in body
    assert "StateDirectory=muneral-kb-sync" in body
    assert "ExecStart=/opt/muneral-kb-sync/current/bin/muneral-kb-sync --incremental --timer" in body


def test_service_hardening_and_retry_are_bounded():
    body = SERVICE.read_text()
    for directive in (
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "NoNewPrivileges=yes",
        "ReadWritePaths=/var/lib/muneral-kb-sync",
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


def _install(root: Path, sha: str, *extra: str, source: Path = REPO, env=None):
    return _run(
        [str(INSTALLER), "install", "--root", str(root), "--sha", sha, "--source", str(source), *extra],
        env=env,
    )


def test_installer_creates_immutable_release_and_atomic_current_previous(tmp_path):
    root = tmp_path / "root"
    first_sha = "1" * 40
    second_sha = "2" * 40
    first = _install(root, first_sha)
    assert first.returncode == 0, first.stdout + first.stderr
    opt = root / "opt/muneral-kb-sync"
    first_release = opt / "releases" / first_sha
    assert (opt / "current").resolve() == first_release.resolve()
    assert not (root / "var/lib/muneral-kb-sync/timer-enabled").exists()
    assert stat.S_IMODE((first_release / "bin/muneral-kb-sync").stat().st_mode) == 0o555
    assert stat.S_IMODE(first_release.stat().st_mode) == 0o555

    second = _install(root, second_sha)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (opt / "current").resolve() == (opt / "releases" / second_sha).resolve()
    assert (opt / "previous").resolve() == first_release.resolve()


def test_installer_rollback_atomically_swaps_current_and_previous(tmp_path):
    root = tmp_path / "root"
    first_sha = "3" * 40
    second_sha = "4" * 40
    assert _install(root, first_sha).returncode == 0
    assert _install(root, second_sha).returncode == 0
    result = _run([str(INSTALLER), "rollback", "--root", str(root)])
    assert result.returncode == 0, result.stdout + result.stderr
    opt = root / "opt/muneral-kb-sync"
    assert (opt / "current").resolve() == (opt / "releases" / first_sha).resolve()
    assert (opt / "previous").resolve() == (opt / "releases" / second_sha).resolve()


def test_installer_requires_explicit_flag_and_pilot_marker_before_timer_enable(tmp_path):
    root = tmp_path / "root"
    sha = "5" * 40
    denied = _install(root, sha, "--enable-timer-after-pilot")
    assert denied.returncode != 0
    state = root / "var/lib/muneral-kb-sync"
    state.mkdir(parents=True, exist_ok=True)
    (state / "pilot-proven").write_text("LTM-0025 pilot graph proven\n")
    still_denied = _install(root, sha, "--enable-timer-after-pilot")
    assert still_denied.returncode != 0
    credentials = root / "etc/muneral-kb-sync"
    for name in ("muneral-db-dsn", "ltm-writer-token"):
        credential = credentials / name
        credential.write_text("test-only-secret\n")
        credential.chmod(0o600)
    allowed = _install(root, sha, "--enable-timer-after-pilot")
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert (state / "timer-enabled").exists()


def test_installer_removes_partial_staging_on_failure(tmp_path):
    root = tmp_path / "root"
    env = {**os.environ, "MUNERAL_KB_SYNC_TEST_FAIL_AFTER_COPY": "1"}
    failed = _install(root, "6" * 40, env=env)
    assert failed.returncode != 0
    release_dir = root / "opt/muneral-kb-sync/releases"
    assert not list(release_dir.glob(".*.tmp.*"))
    assert not (release_dir / ("6" * 40)).exists()


def test_installer_restores_links_and_removes_new_release_on_activation_failure(tmp_path):
    root = tmp_path / "root"
    first_sha = "7" * 40
    failed_sha = "8" * 40
    assert _install(root, first_sha).returncode == 0
    env = {**os.environ, "MUNERAL_KB_SYNC_TEST_FAIL_AFTER_SWITCH": "1"}
    failed = _install(root, failed_sha, env=env)
    assert failed.returncode != 0
    opt = root / "opt/muneral-kb-sync"
    assert (opt / "current").resolve() == (opt / "releases" / first_sha).resolve()
    assert not (opt / "previous").exists()
    assert not (opt / "releases" / failed_sha).exists()
