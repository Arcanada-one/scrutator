"""Provision the Muneral source reader without exposing its credential."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import sys
from pathlib import Path

import asyncpg

ROLE_SQL = Path(__file__).with_name("create_readonly_role.sql")
SET_SECRET_SQL = "SELECT set_config('muneral.role_password', $1, false)"
CLEAR_SECRET_SQL = "SELECT set_config('muneral.role_password', '', false)"


def _read_credential(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("credential path must not be a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("credential path must be a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("credential file mode must be 0600")
    if metadata.st_uid != os.geteuid():
        raise ValueError("credential file must be owned by the effective user")
    value = path.read_text().rstrip("\r\n")
    if not value:
        raise ValueError("credential file must not be empty")
    return value


async def provision_from_files(admin_dsn_file: Path, role_password_file: Path) -> None:
    """Provision over one connection, binding and then clearing the password."""
    admin_dsn = _read_credential(admin_dsn_file)
    role_password = _read_credential(role_password_file)
    connection: asyncpg.Connection | None = None
    setting_bound = False
    try:
        connection = await asyncpg.connect(dsn=admin_dsn)
        await connection.execute(SET_SECRET_SQL, role_password)
        setting_bound = True
        await connection.execute(ROLE_SQL.read_text())
    finally:
        if connection is not None:
            try:
                if setting_bound:
                    await connection.execute(CLEAR_SECRET_SQL)
            finally:
                await connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision the fixed Muneral KB read-only role")
    parser.add_argument("--admin-dsn-file", required=True, type=Path)
    parser.add_argument("--role-password-file", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(provision_from_files(args.admin_dsn_file, args.role_password_file))
    except Exception:
        print("MUNERAL_READONLY_ROLE_PROVISION_FAILED", file=sys.stderr)
        return 1
    print("MUNERAL_READONLY_ROLE_PROVISIONED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
