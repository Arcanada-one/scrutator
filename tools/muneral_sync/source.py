"""Read-only, transactionally consistent Muneral task aggregate adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import asyncpg


@dataclass(frozen=True)
class ChangeRow:
    task_id: str
    revision: int
    changed_at: Any
    deleted: bool


class MuneralSource:
    def __init__(
        self,
        dsn: str,
        *,
        activity_limit: int = 100,
        connect: Callable[..., Awaitable[Any]] = asyncpg.connect,
    ) -> None:
        if not 1 <= activity_limit <= 500:
            raise ValueError("activity_limit must be between 1 and 500")
        self.dsn = dsn
        self.activity_limit = activity_limit
        self._connect = connect
        self._connection: Any | None = None

    async def _get_connection(self):
        if self._connection is None:
            self._connection = await self._connect(self.dsn)
        return self._connection

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def _fetch_task_row(self, conn: Any, task_id: str):
        return await conn.fetchrow(
            """
            SELECT t.id, t.project_id, t.sprint_id, t.parent_id, t.title, t.description,
                   t.status, t.priority, t.due_date, t.estimate_hours, t.created_by_id,
                   t.actor_type, t.created_at, t.updated_at,
                   p.id AS project_identity, p.name AS project_name, p.slug AS project_slug
            FROM tasks t
            LEFT JOIN projects p ON p.id = t.project_id
            WHERE t.id = $1::uuid
            """,
            task_id,
        )

    async def _fetch_related(self, conn: Any, task_id: str) -> dict[str, Any]:
        tags = await conn.fetch("SELECT tag FROM task_tags WHERE task_id = $1::uuid ORDER BY lower(tag), tag", task_id)
        dependencies = await conn.fetch(
            """SELECT type, from_task_id, to_task_id FROM task_dependencies
               WHERE from_task_id = $1::uuid OR to_task_id = $1::uuid
               ORDER BY type, from_task_id, to_task_id""",
            task_id,
        )
        checklists = await conn.fetch(
            """SELECT id, text, checked, position FROM task_checklists
               WHERE task_id = $1::uuid ORDER BY position NULLS LAST, id""",
            task_id,
        )
        agents = await conn.fetch(
            "SELECT agent_id, role FROM task_agents WHERE task_id = $1::uuid ORDER BY agent_id, role",
            task_id,
        )
        activity = await conn.fetch(
            """SELECT id, actor_type, actor_id, action, created_at FROM activity_log
               WHERE task_id = $1::uuid ORDER BY created_at DESC, id DESC LIMIT $2""",
            task_id,
            self.activity_limit,
        )
        return {
            "tags": [item["tag"] for item in tags],
            "dependencies": [dict(item) for item in dependencies],
            "checklists": [dict(item) for item in checklists],
            "agents": [dict(item) for item in agents],
            "activity": [dict(item) for item in reversed(activity)],
        }

    async def fetch_task(self, task_id: str) -> dict[str, Any]:
        conn = await self._get_connection()
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            row = await self._fetch_task_row(conn, task_id)
            if row is None:
                raise LookupError(f"Muneral task not found: {task_id}")
            related = await self._fetch_related(conn, task_id)
        record = dict(row)
        project_aliases = {"project_identity", "project_name", "project_slug"}
        aggregate = {
            "task": {key: record[key] for key in record if key not in project_aliases},
            "project": {
                "id": record.get("project_identity"),
                "name": record.get("project_name"),
                "slug": record.get("project_slug"),
            },
        }
        return {**aggregate, **related}

    async def list_all_task_ids(self) -> list[str]:
        conn = await self._get_connection()
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await conn.fetch("SELECT id FROM tasks ORDER BY id")
        return [str(row["id"]) for row in rows]

    async def list_incremental_changes(self, revisions: dict[str, int]) -> list[ChangeRow]:
        conn = await self._get_connection()
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await conn.fetch(
                """
                SELECT task_id, revision, changed_at, deleted
                FROM muneral_kb_task_changes
                ORDER BY task_id
                """
            )
        changes = [
            ChangeRow(
                task_id=str(row["task_id"]),
                revision=int(row["revision"]),
                changed_at=row["changed_at"],
                deleted=bool(row["deleted"]),
            )
            for row in rows
            if revisions.get(str(row["task_id"])) != int(row["revision"])
        ]
        return changes
