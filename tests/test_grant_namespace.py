"""Tests for scripts/grant_namespace.py — the operator tool that scopes a principal.

`principal_namespace_grants` is the ONLY place a principal's namespace scope is expressed:
`verify_ltm_m2m_token` pins issuer, audience, scope and lifetime, but every M2M reader presents
the same `kb:ltm.read`, so breadth is decided by this table alone. These tests pin the refusals
that keep a grant narrow, because a tool that makes a broad grant easy will eventually make one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "grant_namespace", Path(__file__).resolve().parent.parent / "scripts" / "grant_namespace.py"
)
grant_namespace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(grant_namespace)

Refusal = grant_namespace.Refusal


def _make_pool_mock(mock_conn):
    """asyncpg pool mock with an async context manager (repo convention)."""
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


def _conn(namespace_id=7, grant_present=False, execute_result="INSERT 0 1"):
    conn = AsyncMock()
    conn.fetchval.side_effect = [namespace_id, 1 if grant_present else None]
    conn.execute.return_value = execute_result
    return conn


def _run(argv, conn):
    pool = _make_pool_mock(conn)
    with patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=pool):
        return grant_namespace.main(argv)


# ── narrowness: the refusals that keep a grant from widening ─────────


@pytest.mark.parametrize("value", ["*", "%", "all", "ALL", "any", "everything", "-", ".", "_", ""])
def test_wildcard_namespaces_are_refused(value):
    with pytest.raises(Refusal):
        grant_namespace.validate_namespace(value)


@pytest.mark.parametrize("value", ["wiki,skills", "wiki skills", "wiki;skills", "wiki|skills"])
def test_a_list_of_namespaces_is_refused(value):
    """Two namespaces is two audited commands, never one."""
    with pytest.raises(Refusal) as excinfo:
        grant_namespace.validate_namespace(value)
    assert "exactly one namespace" in str(excinfo.value)


@pytest.mark.parametrize("value", ["*", "all", "", "  "])
def test_wildcard_principals_are_refused(value):
    with pytest.raises(Refusal):
        grant_namespace.validate_principal(value)


@pytest.mark.parametrize("value", ["a/b", "a b", "../etc", "x" * 200, "-lead", "näme"])
def test_malformed_identifiers_are_refused(value):
    with pytest.raises(Refusal):
        grant_namespace.validate_namespace(value)


@pytest.mark.parametrize("value", ["wiki", "self-improvement", "kb.v2", "a", "A9_b-c"])
def test_legitimate_namespace_names_are_accepted(value):
    assert grant_namespace.validate_namespace(value) == value


def test_the_cli_refuses_more_than_one_namespace_per_invocation():
    """argparse must not accept a repeated --namespace as an accumulating list."""
    parser = grant_namespace.build_parser()
    args = parser.parse_args(["--principal", "kb-observer", "--namespace", "wiki", "--namespace", "skills"])
    assert args.namespace == "skills"
    assert not isinstance(args.namespace, list)


# ── the namespace must already exist ─────────────────────────────────


@pytest.mark.asyncio
async def test_granting_an_unknown_namespace_is_refused_not_created():
    conn = AsyncMock()
    conn.fetchval.return_value = None
    with pytest.raises(Refusal) as excinfo:
        await grant_namespace.resolve_namespace_id(conn, "wiki")
    assert "never creates one" in str(excinfo.value)
    conn.execute.assert_not_called()


# ── dry-run is the default ───────────────────────────────────────────


def test_dry_run_is_the_default_and_writes_nothing(capsys):
    conn = _conn()
    assert _run(["--principal", "kb-observer", "--namespace", "wiki"], conn) == 0
    conn.execute.assert_not_called()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "--live" in out


def test_live_applies_exactly_one_insert(capsys):
    conn = _conn()
    assert _run(["--principal", "kb-observer", "--namespace", "wiki", "--live"], conn) == 0
    assert conn.execute.await_count == 1
    sql, principal, namespace_id = conn.execute.await_args.args
    assert sql.startswith("INSERT INTO principal_namespace_grants")
    assert (principal, namespace_id) == ("kb-observer", 7)
    assert "applied" in capsys.readouterr().out


def test_regranting_is_idempotent_and_reported_as_no_change(capsys):
    conn = _conn(grant_present=True, execute_result="INSERT 0 0")
    assert _run(["--principal", "kb-observer", "--namespace", "wiki", "--live"], conn) == 0
    assert "no change" in capsys.readouterr().out


# ── revocation ───────────────────────────────────────────────────────


def test_revoke_deletes_exactly_one_grant(capsys):
    conn = _conn(grant_present=True, execute_result="DELETE 1")
    assert _run(["--principal", "kb-observer", "--namespace", "wiki", "--revoke", "--live"], conn) == 0
    sql, principal, namespace_id = conn.execute.await_args.args
    assert sql.startswith("DELETE FROM principal_namespace_grants")
    assert (principal, namespace_id) == ("kb-observer", 7)
    assert "applied" in capsys.readouterr().out


def test_revoking_an_absent_grant_is_a_no_op(capsys):
    conn = _conn(grant_present=False, execute_result="DELETE 0")
    assert _run(["--principal", "kb-observer", "--namespace", "wiki", "--revoke", "--live"], conn) == 0
    assert "no change" in capsys.readouterr().out


# ── audit ────────────────────────────────────────────────────────────


def test_list_is_read_only(capsys):
    conn = AsyncMock()
    conn.fetch.return_value = [{"principal_id": "kb-observer", "namespace": "wiki", "granted_at": "t"}]
    assert _run(["--list"], conn) == 0
    conn.execute.assert_not_called()
    assert "kb-observer" in capsys.readouterr().out


def test_list_reports_the_empty_deny_all_state(capsys):
    conn = AsyncMock()
    conn.fetch.return_value = []
    assert _run(["--list"], conn) == 0
    assert "denied" in capsys.readouterr().out


def test_list_refuses_to_be_combined_with_a_mutation():
    assert grant_namespace.main(["--list", "--principal", "kb-observer", "--namespace", "wiki", "--live"]) == 2


# ── usage ────────────────────────────────────────────────────────────


def test_missing_arguments_exit_2():
    assert grant_namespace.main(["--principal", "kb-observer"]) == 2
    assert grant_namespace.main([]) == 2


def test_a_refusal_exits_1_and_never_writes(capsys):
    conn = _conn()
    assert _run(["--principal", "kb-observer", "--namespace", "*", "--live"], conn) == 1
    conn.execute.assert_not_called()
    assert "REFUSED" in capsys.readouterr().err
