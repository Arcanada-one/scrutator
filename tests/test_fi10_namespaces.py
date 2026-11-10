"""Synthetic FI10 namespace ACL, readback and withdrawal purge (FI10-T5.2)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from scrutator.fi10.harness import (
    PRIVATE_RECORD_ID,
    PUBLIC_RECORD_ID,
    SYNTHETIC_GENERATION,
    WITHDRAW_RECORD_ID,
    build_seeded_store,
    run_harness,
    write_receipt,
)
from scrutator.fi10.namespaces import FI10_PRIVATE, FI10_PUBLIC, decide_read
from scrutator.fi10.readback import readback_generation, scoped_query
from scrutator.fi10.withdrawal import purge_withdrawn, resolve_serving_generation

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_acl_blocks_private_disclosure_in_public_namespace():
    decision = decide_read(
        principal_id="reader",
        granted_namespaces={FI10_PUBLIC, FI10_PRIVATE},
        namespace=FI10_PUBLIC,
        disclosure_class="private",
    )
    assert decision.verdict == "deny"
    assert decision.reason == "private_disclosure_blocked_in_public"


def test_acl_requires_exact_namespace_grant():
    decision = decide_read(
        principal_id="reader",
        granted_namespaces={FI10_PUBLIC},
        namespace=FI10_PRIVATE,
        disclosure_class="private",
    )
    assert decision.verdict == "deny"
    assert decision.reason == "namespace_not_granted"


def test_positive_control_and_zero_private_in_public():
    store = build_seeded_store()
    readback = readback_generation(
        store,
        generation_id=SYNTHETIC_GENERATION,
        principal_id="fi10-public-reader",
        granted_namespaces=frozenset({FI10_PUBLIC}),
        positive_control_record_id=PUBLIC_RECORD_ID,
    )
    assert readback.positive_control_found is True
    assert readback.private_records_in_public == 0
    assert readback.generation_id == SYNTHETIC_GENERATION
    assert readback.database == "synthetic-fi10"
    assert readback.collection == "scrutator-fi10"
    public = next(n for n in readback.namespaces if n.namespace == FI10_PUBLIC)
    assert public.vector_hits >= 1
    assert public.typed_graph_hits >= 1


def test_private_record_not_in_public_but_found_under_private_acl():
    store = build_seeded_store()
    public_hits = scoped_query(
        store,
        principal_id="pub",
        granted_namespaces=frozenset({FI10_PUBLIC}),
        namespace=FI10_PUBLIC,
        generation_id=SYNTHETIC_GENERATION,
    )
    private_hits = scoped_query(
        store,
        principal_id="priv",
        granted_namespaces=frozenset({FI10_PRIVATE}),
        namespace=FI10_PRIVATE,
        generation_id=SYNTHETIC_GENERATION,
    )
    assert PRIVATE_RECORD_ID not in public_hits
    assert PRIVATE_RECORD_ID in private_hits


def test_withdrawal_purge_removes_search_entries():
    store = build_seeded_store()
    result = purge_withdrawn(store, record_id=WITHDRAW_RECORD_ID, generation_id=SYNTHETIC_GENERATION)
    assert result.purged is True
    assert result.search_entries_remaining == 0
    remaining = scoped_query(
        store,
        principal_id="pub",
        granted_namespaces=frozenset({FI10_PUBLIC}),
        namespace=FI10_PUBLIC,
        generation_id=SYNTHETIC_GENERATION,
    )
    assert WITHDRAW_RECORD_ID not in remaining


def test_revoked_generation_never_falls_back_to_lkg():
    store = build_seeded_store()
    ok = resolve_serving_generation(store, requested_generation=None, projection_failed=True)
    assert ok.verdict == "serve"
    assert ok.generation_id == SYNTHETIC_GENERATION
    store.mark_revoked(SYNTHETIC_GENERATION)
    refused = resolve_serving_generation(store, requested_generation=None, projection_failed=True)
    assert refused.verdict == "refuse"
    assert refused.reason == "revoked_generation_forbidden_fallback"


def test_harness_all_criteria_pass(tmp_path: Path):
    receipt = write_receipt(tmp_path / "receipt.json")
    assert receipt["all_pass"] is True
    assert receipt["runtime_authorized"] is False
    assert receipt["knowledge_admitted"] is False
    assert receipt["live_namespace_enablement"] is False
    assert all(receipt["criteria"].values())
    on_disk = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert on_disk["generation_id"] == SYNTHETIC_GENERATION
    assert len(on_disk["receipt_sha256"]) == 64
    assert run_harness()["all_pass"] is True


def test_cli_refuses_live(capsys):
    cli = _load_script("fi10_namespace_readback.py")
    assert cli.main(["--live"]) == 1
    assert "live_enablement_held" in capsys.readouterr().err


def test_bootstrap_preview_and_live_refuse(capsys):
    boot = _load_script("bootstrap_fi10_namespaces.py")
    assert boot.main([]) == 0
    assert "FI10_NAMESPACE_BOOTSTRAP_PREVIEW" in capsys.readouterr().out
    assert boot.main(["--live"]) == 1
    assert "live_enablement_held" in capsys.readouterr().err


def test_compose_feeder_grant_not_widened_for_fi10():
    """No live enablement: production compose must not yet list fi10-* feeder scopes."""
    import yaml

    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    grant = compose["services"]["scrutator"]["environment"]["SCRUTATOR_FEEDER_NAMESPACES"]
    assert "fi10-public" not in grant
    assert "fi10-private" not in grant
