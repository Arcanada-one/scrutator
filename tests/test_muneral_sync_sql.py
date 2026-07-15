from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit

import asyncpg
import pytest

from tools.muneral_sync.sql.provision_readonly_role import provision_from_files

ROOT = Path(__file__).parents[1]
SQL_DIR = ROOT / "tools" / "muneral_sync" / "sql"
SEED = SQL_DIR / "pilot_seed.sql"
ROLLBACK = SQL_DIR / "pilot_rollback.sql"
READONLY_ROLE = SQL_DIR / "create_readonly_role.sql"

PILOT_USER_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000001"
PILOT_WORKSPACE_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000002"
PILOT_PROJECT_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000003"
PILOT_TASK_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000004"
PILOT_CHECKLIST_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000005"
PILOT_ACTIVITY_ID = "7d2c0e8a-4b7e-4c01-8d33-100000000006"


def test_pilot_seed_is_deterministic_idempotent_and_complete() -> None:
    sql = SEED.read_text()

    uuid_literals = set(re.findall(r"[0-9a-f]{8}-[0-9a-f-]{27,}", sql))
    expected = {
        PILOT_USER_ID,
        PILOT_WORKSPACE_ID,
        PILOT_PROJECT_ID,
        PILOT_TASK_ID,
        PILOT_CHECKLIST_ID,
        PILOT_ACTIVITY_ID,
    }
    assert expected <= uuid_literals
    assert all(uuid.UUID(value).version == 4 for value in expected)
    assert sql.count("ON CONFLICT DO NOTHING") == 8
    assert "LTM-0025 Muneral to KB graph-merge pilot" in sql
    assert "Long Term Memory" in sql
    assert "'agent'" in sql
    assert "'graph-merge'" in sql
    assert "'pilot'" in sql
    assert "task_checklists" in sql
    assert "activity_log" in sql
    assert "pilot seed identity mismatch" in sql


def test_pilot_rollback_is_fk_safe_scoped_and_preserves_registry_tombstone() -> None:
    sql = ROLLBACK.read_text()
    delete_order = [
        "activity_log",
        "task_checklists",
        "task_tags",
        "tasks",
        "projects",
        "workspaces",
        "users",
    ]
    offsets = [sql.index(f"DELETE FROM {table}") for table in delete_order]

    assert offsets == sorted(offsets)
    assert "DELETE FROM muneral_kb_task_changes" not in sql
    assert "7d2c0e8a-4b7e-4c01-8d33-10000000000" in sql
    assert "WHERE id =" in sql
    assert "pilot rollback identity mismatch" in sql
    assert "non-pilot dependents" in sql
    assert sql.count("IS NOT DISTINCT FROM") >= 10


def test_readonly_role_is_fail_closed_and_documents_database_wide_revocations() -> None:
    sql = READONLY_ROLE.read_text()

    assert "current_setting('muneral.role_password', true)" in sql
    assert "muneral_kb_reader" in sql
    assert "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS" in sql
    assert "default_transaction_read_only = on" in sql
    assert "statement_timeout = '15s'" in sql
    assert "REVOKE TEMPORARY ON DATABASE" in sql
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC" in sql
    assert "REVOKE CREATE ON DATABASE" in sql
    assert "database-wide" in sql.lower()
    assert "existing application roles" in sql.lower()

    granted_tables = set(re.findall(r"GRANT SELECT ON TABLE public\.([a-z_]+)", sql))
    assert granted_tables == {
        "users",
        "workspaces",
        "projects",
        "tasks",
        "task_tags",
        "task_checklists",
        "task_dependencies",
        "task_agents",
        "agents",
        "activity_log",
        "muneral_kb_task_changes",
    }
    assert "GRANT SELECT ON ALL TABLES" not in sql
    assert "GRANT ALL" not in sql
    assert "SELECT set_config('muneral.role_password', '<secret>'" not in sql
    assert "SELECT set_config('muneral.role_password', '', false)" not in sql


def _muneral_migrations() -> list[Path]:
    default_repo = ROOT.parents[1] / "Muneral" / "code"
    repo = Path(os.environ.get("MUNERAL_REPO", default_repo))
    migrations = repo / "apps" / "api" / "prisma" / "migrations"
    return [
        migrations / "20260607103126_init_schema" / "migration.sql",
        migrations / "20260607200000_add_field_tracking" / "migration.sql",
        migrations / "20260715170000_add_muneral_kb_task_changes" / "migration.sql",
    ]


async def _assert_permission_denied(connection: asyncpg.Connection, sql: str) -> None:
    with pytest.raises((asyncpg.InsufficientPrivilegeError, asyncpg.ReadOnlySQLTransactionError)):
        await connection.execute(sql)


@pytest.mark.asyncio
async def test_pilot_sql_against_disposable_pg18_with_live_due_date_signature(tmp_path: Path) -> None:
    dsn = os.environ.get("MUNERAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("no disposable PG18 MUNERAL_TEST_DATABASE_URL; raw-SQL gate is environment-gated")

    admin = await asyncpg.connect(dsn)
    reader: asyncpg.Connection | None = None
    try:
        database = await admin.fetchval("SELECT current_database()")
        assert database.startswith("ltm0025_test_"), "refuse SQL integration test outside disposable database"
        assert await admin.fetchval("SHOW server_version_num") >= "180000"

        for migration in _muneral_migrations():
            await admin.execute(migration.read_text())
        # The committed initial migration is stale (TEXT), while the 2026-07-15
        # live Muneral schema probe reported tasks.due_date as DATE. Reproduce
        # and assert that exact live signature without touching production.
        await admin.execute("ALTER TABLE tasks ALTER COLUMN due_date TYPE DATE USING due_date::date")
        assert (
            await admin.fetchval(
                """SELECT data_type FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'tasks' AND column_name = 'due_date'"""
            )
            == "date"
        )

        # A deterministic-ID collision must abort the whole seed and preserve
        # the pre-existing row without creating any pilot descendants.
        await admin.execute(
            """INSERT INTO users (id, name, created_at, updated_at)
            VALUES ($1, 'Collision fixture', '2026-07-15T00:00:00Z', '2026-07-15T00:00:00Z')""",
            PILOT_USER_ID,
        )
        with pytest.raises(asyncpg.RaiseError, match="pilot seed identity mismatch"):
            await admin.execute(SEED.read_text())
        await admin.execute("ROLLBACK")
        assert await admin.fetchval("SELECT name FROM users WHERE id = $1", PILOT_USER_ID) == "Collision fixture"
        assert await admin.fetchval("SELECT count(*) FROM workspaces WHERE id = $1", PILOT_WORKSPACE_ID) == 0
        await admin.execute("DELETE FROM users WHERE id = $1", PILOT_USER_ID)

        unrelated_user = "d96f5d01-4e1e-4b00-9000-000000000001"
        unrelated_workspace = "d96f5d01-4e1e-4b00-9000-000000000002"
        unrelated_project = "d96f5d01-4e1e-4b00-9000-000000000003"
        unrelated_task = "d96f5d01-4e1e-4b00-9000-000000000004"
        async with admin.transaction():
            await admin.execute(
                """INSERT INTO users (id, name, created_at, updated_at)
                VALUES ($1, 'Unrelated fixture', '2026-07-15T00:00:00Z', '2026-07-15T00:00:00Z')""",
                unrelated_user,
            )
            await admin.execute(
                """INSERT INTO workspaces (id, slug, name, owner_id, created_at)
                VALUES ($1, 'unrelated-fixture', 'Unrelated fixture', $2, '2026-07-15T00:00:00Z')""",
                unrelated_workspace,
                unrelated_user,
            )
            await admin.execute(
                """INSERT INTO projects (id, workspace_id, slug, name, created_at)
                VALUES ($1, $2, 'unrelated-fixture', 'Unrelated fixture', '2026-07-15T00:00:00Z')""",
                unrelated_project,
                unrelated_workspace,
            )
            await admin.execute(
                """INSERT INTO tasks (id, project_id, title, updated_at)
                VALUES ($1, $2, 'Unrelated fixture', '2026-07-15T00:00:00Z')""",
                unrelated_task,
                unrelated_project,
            )
            await admin.execute("INSERT INTO task_tags (task_id, tag) VALUES ($1, 'unrelated')", unrelated_task)
            await admin.execute(
                "INSERT INTO task_checklists (task_id, text) VALUES ($1, 'Unrelated fixture')", unrelated_task
            )
            await admin.execute(
                """INSERT INTO activity_log (task_id, workspace_id, actor_type, actor_id, action)
                VALUES ($1, $2, 'agent', $3, 'task.created')""",
                unrelated_task,
                unrelated_workspace,
                unrelated_user,
            )

        await admin.execute(SEED.read_text())
        first_counts = await admin.fetchrow(
            """
            SELECT
              (SELECT count(*) FROM users WHERE id = $1) AS users,
              (SELECT count(*) FROM workspaces WHERE id = $2) AS workspaces,
              (SELECT count(*) FROM projects WHERE id = $3) AS projects,
              (SELECT count(*) FROM tasks WHERE id = $4) AS tasks,
              (SELECT count(*) FROM task_tags WHERE task_id = $4) AS tags,
              (SELECT count(*) FROM task_checklists WHERE task_id = $4) AS checklists,
              (SELECT count(*) FROM activity_log WHERE task_id = $4) AS activities
            """,
            PILOT_USER_ID,
            PILOT_WORKSPACE_ID,
            PILOT_PROJECT_ID,
            PILOT_TASK_ID,
        )
        assert tuple(first_counts) == (1, 1, 1, 1, 2, 1, 1)

        await admin.execute(SEED.read_text())
        second_counts = await admin.fetchrow(
            """
            SELECT
              (SELECT count(*) FROM users WHERE id = $1),
              (SELECT count(*) FROM workspaces WHERE id = $2),
              (SELECT count(*) FROM projects WHERE id = $3),
              (SELECT count(*) FROM tasks WHERE id = $4),
              (SELECT count(*) FROM task_tags WHERE task_id = $4),
              (SELECT count(*) FROM task_checklists WHERE task_id = $4),
              (SELECT count(*) FROM activity_log WHERE task_id = $4)
            """,
            PILOT_USER_ID,
            PILOT_WORKSPACE_ID,
            PILOT_PROJECT_ID,
            PILOT_TASK_ID,
        )
        assert tuple(second_counts) == tuple(first_counts)

        # Identity drift blocks rollback before its first DELETE.
        await admin.execute("UPDATE task_checklists SET text = 'Collision fixture' WHERE id = $1", PILOT_CHECKLIST_ID)
        with pytest.raises(asyncpg.RaiseError, match="pilot rollback identity mismatch"):
            await admin.execute(ROLLBACK.read_text())
        await admin.execute("ROLLBACK")
        assert await admin.fetchval("SELECT count(*) FROM tasks WHERE id = $1", PILOT_TASK_ID) == 1
        assert await admin.fetchval("SELECT count(*) FROM activity_log WHERE id = $1", PILOT_ACTIVITY_ID) == 1
        assert await admin.fetchval("SELECT text FROM task_checklists WHERE id = $1", PILOT_CHECKLIST_ID) == (
            "Collision fixture"
        )
        await admin.execute(
            """UPDATE task_checklists
            SET text = 'Pilot task graph is proven in the KB through the granted principal'
            WHERE id = $1""",
            PILOT_CHECKLIST_ID,
        )

        supplied_dsn_file = os.environ.get("MUNERAL_TEST_ADMIN_DSN_FILE")
        supplied_password_file = os.environ.get("MUNERAL_TEST_ROLE_PASSWORD_FILE")
        dsn_file = Path(supplied_dsn_file) if supplied_dsn_file else tmp_path / "admin-dsn"
        password_file = Path(supplied_password_file) if supplied_password_file else tmp_path / "reader-password"
        if not supplied_dsn_file:
            dsn_file.write_text(dsn + "\n")
            dsn_file.chmod(0o600)
        if not supplied_password_file:
            password_file.write_text(uuid.uuid4().hex + "\n")
            password_file.chmod(0o600)
        role_password = password_file.read_text().strip()
        await provision_from_files(dsn_file, password_file)

        parsed = urlsplit(dsn)
        reader = await asyncpg.connect(
            host=parsed.hostname,
            port=parsed.port,
            user="muneral_kb_reader",
            password=role_password,
            database=unquote(parsed.path.removeprefix("/")),
        )
        assert await reader.fetchval("SELECT title FROM tasks WHERE id = $1", PILOT_TASK_ID) == (
            "LTM-0025 Muneral to KB graph-merge pilot"
        )
        assert await reader.fetchval("SHOW default_transaction_read_only") == "on"
        assert await reader.fetchval("SHOW statement_timeout") == "15s"
        # The default is operational defence-in-depth, not an authorization
        # boundary. Prove the grants still deny writes if a client flips it.
        await reader.execute("SET default_transaction_read_only = off")
        await _assert_permission_denied(reader, "INSERT INTO tasks (id) VALUES (gen_random_uuid())")
        await _assert_permission_denied(reader, "CREATE TABLE public.reader_escape (id integer)")
        await _assert_permission_denied(reader, "CREATE TEMP TABLE reader_escape (id integer)")
        await _assert_permission_denied(reader, "SELECT * FROM task_field_state")

        # A later non-pilot child makes rollback fail closed. Because the
        # rollback is one transaction, none of the pilot graph is deleted.
        later_child_id = "d96f5d01-4e1e-4b00-9000-000000000005"
        await admin.execute(
            "INSERT INTO task_git_refs (id, task_id, type, url) VALUES ($1, $2, 'commit', 'https://example.invalid')",
            later_child_id,
            PILOT_TASK_ID,
        )
        with pytest.raises(asyncpg.RaiseError, match="non-pilot dependents"):
            await admin.execute(ROLLBACK.read_text())
        await admin.execute("ROLLBACK")
        assert await admin.fetchval("SELECT count(*) FROM tasks WHERE id = $1", PILOT_TASK_ID) == 1
        assert await admin.fetchval("SELECT count(*) FROM activity_log WHERE id = $1", PILOT_ACTIVITY_ID) == 1
        assert await admin.fetchval("SELECT count(*) FROM task_git_refs WHERE id = $1", later_child_id) == 1
        await admin.execute("DELETE FROM task_git_refs WHERE id = $1", later_child_id)

        await admin.execute(ROLLBACK.read_text())
        assert await admin.fetchval("SELECT count(*) FROM tasks WHERE id = $1", PILOT_TASK_ID) == 0
        assert await admin.fetchval("SELECT count(*) FROM users WHERE id = $1", unrelated_user) == 1
        assert await admin.fetchval("SELECT count(*) FROM tasks WHERE id = $1", unrelated_task) == 1
        assert await admin.fetchval("SELECT count(*) FROM task_tags WHERE task_id = $1", unrelated_task) == 1
        assert await admin.fetchval("SELECT count(*) FROM task_checklists WHERE task_id = $1", unrelated_task) == 1
        assert await admin.fetchval("SELECT count(*) FROM activity_log WHERE task_id = $1", unrelated_task) == 1
        tombstone = await admin.fetchrow(
            "SELECT revision, deleted FROM muneral_kb_task_changes WHERE task_id = $1", PILOT_TASK_ID
        )
        assert tombstone["revision"] > 1
        assert tombstone["deleted"] is True
    finally:
        if reader is not None:
            await reader.close()
        await admin.close()
