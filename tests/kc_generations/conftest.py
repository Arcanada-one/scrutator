"""Native PostgreSQL fixtures. Tests need a DISPOSABLE PostgreSQL with pgvector:

    KC_TEST_PG_ADMIN_DSN=postgresql://postgres@127.0.0.1:<port>/postgres   (superuser)

Without it every test here is skipped (reported as NOT_MEASURED, never as a pass). A fresh database is
created per session through the repository's REAL installer (``apply_schema``) and removed afterwards.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest
import pytest_asyncio

from scrutator.config import settings
from scrutator.db import connection
from scrutator.db import kc_generations as kc
from tests.kc_generations.helpers import SyntheticAuthority, make_world

ADMIN_DSN = os.environ.get("KC_TEST_PG_ADMIN_DSN")
_REMOVE = "DROP DATABASE IF EXISTS"  # removes only the disposable per-session test database created above


def _dsn_for(db: str) -> str:
    return f"{ADMIN_DSN.rsplit('/', 1)[0]}/{db}"


@pytest.fixture(scope="session")
def kc_dsn():
    if not ADMIN_DSN:
        pytest.skip("KC_TEST_PG_ADMIN_DSN not set: native PostgreSQL tests are NOT_MEASURED")
    db = f"kcgen_{uuid.uuid4().hex[:10]}"

    async def admin(sql: str) -> None:
        c = await asyncpg.connect(ADMIN_DSN)
        try:
            await c.execute(sql)
        finally:
            await c.close()

    async def install() -> None:
        # a pool registers the pgvector codec on connect, so the extension must pre-exist (as in production)
        c = await asyncpg.connect(_dsn_for(db))
        try:
            await c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await c.close()
        settings.database_url = _dsn_for(db)
        connection._pool = None
        await connection.apply_schema()  # the repository's real installer
        await connection.close_pool()

    asyncio.run(admin(f'CREATE DATABASE "{db}"'))
    asyncio.run(install())
    yield _dsn_for(db)
    asyncio.run(admin(f'{_REMOVE} "{db}" WITH (FORCE)'))


@pytest_asyncio.fixture
async def pool(kc_dsn):
    p = await asyncpg.create_pool(kc_dsn, min_size=2, max_size=16, init=connection._init_connection)
    connection._pool = p
    kc.configure_authority_backend(None)
    try:
        yield p
    finally:
        connection._pool = None
        kc.configure_authority_backend(None)
        await p.close()


@pytest_asyncio.fixture
async def world(pool):
    return await make_world(pool)


@pytest.fixture
def authority():
    return SyntheticAuthority({"p1"})
