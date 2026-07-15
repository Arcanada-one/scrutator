from __future__ import annotations

import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tools.muneral_sync.sql import provision_readonly_role

SQL_DIR = Path(__file__).parents[1] / "tools" / "muneral_sync" / "sql"
RUNNER = SQL_DIR / "run_pilot_pg18_smoke.sh"


def _credential(path: Path, value: str, mode: int = 0o600) -> Path:
    path.write_text(value + "\n")
    path.chmod(mode)
    return path


def test_provisioner_uses_bound_secret_and_emits_only_success_marker(tmp_path, monkeypatch, capsys) -> None:
    dsn_secret = "dsn-secret-sentinel"
    role_secret = "role-secret-sentinel"
    dsn_file = _credential(tmp_path / "admin-dsn", f"postgresql://admin:{dsn_secret}@db/test")
    password_file = _credential(tmp_path / "reader-password", role_secret)
    connection = AsyncMock()
    monkeypatch.setattr(provision_readonly_role.asyncpg, "connect", AsyncMock(return_value=connection))
    argv = ["--admin-dsn-file", str(dsn_file), "--role-password-file", str(password_file)]

    assert provision_readonly_role.main(argv) == 0

    output = capsys.readouterr()
    assert output.out == "MUNERAL_READONLY_ROLE_PROVISIONED\n"
    assert output.err == ""
    assert dsn_secret not in output.out + output.err + " ".join(argv)
    assert role_secret not in output.out + output.err + " ".join(argv)
    set_call, sql_call, clear_call = connection.execute.await_args_list
    assert set_call.args == ("SELECT set_config('muneral.role_password', $1, false)", role_secret)
    assert role_secret not in sql_call.args[0]
    assert clear_call.args == ("SELECT set_config('muneral.role_password', '', false)",)
    connection.close.assert_awaited_once()


def test_provisioner_suppresses_driver_error_secrets_and_clears_setting(tmp_path, monkeypatch, capsys) -> None:
    dsn_secret = "dsn-error-secret"
    role_secret = "role-error-secret"
    dsn_file = _credential(tmp_path / "admin-dsn", f"postgresql://admin:{dsn_secret}@db/test")
    password_file = _credential(tmp_path / "reader-password", role_secret)
    connection = AsyncMock()
    connection.execute.side_effect = [None, RuntimeError(f"driver leaked {role_secret}"), None]
    monkeypatch.setattr(provision_readonly_role.asyncpg, "connect", AsyncMock(return_value=connection))
    argv = ["--admin-dsn-file", str(dsn_file), "--role-password-file", str(password_file)]

    assert provision_readonly_role.main(argv) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "MUNERAL_READONLY_ROLE_PROVISION_FAILED\n"
    assert dsn_secret not in output.out + output.err + " ".join(argv)
    assert role_secret not in output.out + output.err + " ".join(argv)
    assert connection.execute.await_args_list[-1].args == ("SELECT set_config('muneral.role_password', '', false)",)
    connection.close.assert_awaited_once()


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644, 0o660])
def test_provisioner_rejects_credential_files_that_are_not_mode_0600(tmp_path, monkeypatch, capsys, mode) -> None:
    dsn_file = _credential(tmp_path / "admin-dsn", "postgresql://admin:secret@db/test", mode=mode)
    password_file = _credential(tmp_path / "reader-password", "reader-secret")
    connect = AsyncMock()
    monkeypatch.setattr(provision_readonly_role.asyncpg, "connect", connect)

    assert (
        provision_readonly_role.main(["--admin-dsn-file", str(dsn_file), "--role-password-file", str(password_file)])
        == 1
    )

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "MUNERAL_READONLY_ROLE_PROVISION_FAILED\n"
    connect.assert_not_awaited()
    assert stat.S_IMODE(dsn_file.stat().st_mode) == mode


def test_provisioner_cli_has_no_secret_value_arguments() -> None:
    parser = provision_readonly_role.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert option_strings == {"-h", "--help", "--admin-dsn-file", "--role-password-file"}
    assert not {"--dsn", "--password", "--secret"} & option_strings
    assert "secret" not in " ".join(sys.argv)


def test_pg18_runner_is_failure_safe_and_has_explicit_residue_checks() -> None:
    script = RUNNER.read_text()

    assert "trap cleanup EXIT" in script
    assert "DROP DATABASE IF EXISTS" in script
    assert "DROP ROLE IF EXISTS muneral_kb_reader" in script
    assert 'docker_on_target rm -f "$container"' in script
    assert "--inject-failure" in script
    assert "PG18_SMOKE_ZERO_RESIDUE" in script
    assert "MUNERAL_TEST_DATABASE_URL" in script
    assert "chmod 600" in script
