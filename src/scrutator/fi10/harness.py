"""Synthetic end-to-end harness for FI10-T5.2 acceptance controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scrutator.fi10.namespaces import FI10_PRIVATE, FI10_PUBLIC, decide_read
from scrutator.fi10.readback import readback_generation, scoped_query
from scrutator.fi10.store import Fi10Record, Fi10Relation, SyntheticFi10Store
from scrutator.fi10.withdrawal import purge_withdrawn, resolve_serving_generation

SYNTHETIC_GENERATION = "fi10-public-2026-11-10-1-synthetic"
PUBLIC_PRINCIPAL = "fi10-public-reader"
PRIVATE_PRINCIPAL = "fi10-private-reader"
PUBLIC_RECORD_ID = "syn-public-occupation-1"
PRIVATE_RECORD_ID = "syn-private-actor-1"
WITHDRAW_RECORD_ID = "syn-public-withdraw-1"


def build_seeded_store() -> SyntheticFi10Store:
    store = SyntheticFi10Store()
    store.upsert_record(
        Fi10Record(
            record_id=PUBLIC_RECORD_ID,
            namespace=FI10_PUBLIC,
            disclosure_class="public",
            generation_id=SYNTHETIC_GENERATION,
            body="SYNTHETIC public occupation card for FI10-T5.2 positive control.",
        )
    )
    store.upsert_record(
        Fi10Record(
            record_id=WITHDRAW_RECORD_ID,
            namespace=FI10_PUBLIC,
            disclosure_class="public",
            generation_id=SYNTHETIC_GENERATION,
            body="SYNTHETIC public disclosure scheduled for withdrawal purge proof.",
        )
    )
    store.upsert_record(
        Fi10Record(
            record_id=PRIVATE_RECORD_ID,
            namespace=FI10_PRIVATE,
            disclosure_class="private",
            generation_id=SYNTHETIC_GENERATION,
            body="SYNTHETIC private actor binding — must never appear in fi10-public.",
        )
    )
    store.upsert_relation(
        Fi10Relation(
            edge_id="syn-edge-requires-1",
            namespace=FI10_PUBLIC,
            generation_id=SYNTHETIC_GENERATION,
            source_record_id=PUBLIC_RECORD_ID,
            target_record_id=WITHDRAW_RECORD_ID,
            relation="requires",
        )
    )
    store.last_known_good_generation = SYNTHETIC_GENERATION
    store.active_generation = SYNTHETIC_GENERATION
    return store


def run_harness() -> dict[str, Any]:
    """Execute positive/negative controls and return a machine receipt."""
    store = build_seeded_store()
    public_grants = frozenset({FI10_PUBLIC})
    private_grants = frozenset({FI10_PRIVATE})
    both_grants = frozenset({FI10_PUBLIC, FI10_PRIVATE})

    # AC1 / AC3: scoped public query with positive control; zero private in public.
    readback_before = readback_generation(
        store,
        generation_id=SYNTHETIC_GENERATION,
        principal_id=PUBLIC_PRINCIPAL,
        granted_namespaces=public_grants,
        positive_control_record_id=PUBLIC_RECORD_ID,
    )
    ac1 = (
        readback_before.positive_control_found
        and readback_before.private_records_in_public == 0
        and PUBLIC_RECORD_ID in scoped_query(
            store,
            principal_id=PUBLIC_PRINCIPAL,
            granted_namespaces=public_grants,
            namespace=FI10_PUBLIC,
            generation_id=SYNTHETIC_GENERATION,
        )
    )

    # ACL negatives: private record not in public; found under private ACL.
    private_in_public = scoped_query(
        store,
        principal_id=PUBLIC_PRINCIPAL,
        granted_namespaces=public_grants,
        namespace=FI10_PUBLIC,
        generation_id=SYNTHETIC_GENERATION,
    )
    private_under_acl = scoped_query(
        store,
        principal_id=PRIVATE_PRINCIPAL,
        granted_namespaces=private_grants,
        namespace=FI10_PRIVATE,
        generation_id=SYNTHETIC_GENERATION,
    )
    leak_decision = decide_read(
        principal_id=PUBLIC_PRINCIPAL,
        granted_namespaces=both_grants,
        namespace=FI10_PUBLIC,
        disclosure_class="private",
    )

    # AC2: vector + typed-graph measured for same generation; withdrawal purge.
    public_ns = next(n for n in readback_before.namespaces if n.namespace == FI10_PUBLIC)
    same_generation_projections = (
        public_ns.vector_hits >= 1
        and public_ns.typed_graph_hits >= 1
        and readback_before.generation_id == SYNTHETIC_GENERATION
    )
    purge = purge_withdrawn(store, record_id=WITHDRAW_RECORD_ID, generation_id=SYNTHETIC_GENERATION)
    after_public = scoped_query(
        store,
        principal_id=PUBLIC_PRINCIPAL,
        granted_namespaces=public_grants,
        namespace=FI10_PUBLIC,
        generation_id=SYNTHETIC_GENERATION,
    )
    withdrawn_absent = WITHDRAW_RECORD_ID not in after_public and purge.search_entries_remaining == 0

    # LKG on projection failure; revoked generation never falls back.
    lkg = resolve_serving_generation(store, requested_generation=None, projection_failed=True)
    store.mark_revoked(SYNTHETIC_GENERATION)
    revoked_refuse = resolve_serving_generation(
        store, requested_generation=SYNTHETIC_GENERATION, projection_failed=False
    )
    revoked_lkg_refuse = resolve_serving_generation(store, requested_generation=None, projection_failed=True)

    # Manifest ↔ index generation parity (synthetic manifest slice).
    manifest = store.to_manifest_slice(SYNTHETIC_GENERATION)
    generation_parity = manifest["generation_id"] == readback_before.generation_id

    criteria = {
        "AC1_zero_private_in_public_with_positive_control": ac1,
        "AC2_vector_graph_same_generation_and_purge": same_generation_projections and withdrawn_absent and lkg.verdict == "serve",
        "AC3_positive_control_generation_parity": readback_before.positive_control_found and generation_parity,
        "AC4_negatives_private_acl_withdrawn_forbidden_fallback": (
            PRIVATE_RECORD_ID not in private_in_public
            and PRIVATE_RECORD_ID in private_under_acl
            and withdrawn_absent
            and leak_decision.verdict == "deny"
            and revoked_refuse.verdict == "refuse"
            and revoked_lkg_refuse.verdict == "refuse"
            and revoked_refuse.reason == "revoked_generation_forbidden_fallback"
        ),
    }

    receipt: dict[str, Any] = {
        "schema": "Fi10NamespaceHarnessReceipt/v1",
        "label": "synthetic",
        "task": "FI10-T5.2",
        "generation_id": SYNTHETIC_GENERATION,
        "runtime_authorized": False,
        "knowledge_admitted": False,
        "live_namespace_enablement": False,
        "production_purge": False,
        "readback_before": readback_before.to_dict(),
        "purge": purge.to_dict(),
        "serving": {
            "lkg_on_projection_failure": lkg.to_dict(),
            "revoked_direct": revoked_refuse.to_dict(),
            "revoked_lkg": revoked_lkg_refuse.to_dict(),
        },
        "manifest_slice": manifest,
        "criteria": criteria,
        "all_pass": all(criteria.values()),
    }
    return receipt


def write_receipt(path: Path) -> dict[str, Any]:
    receipt = run_harness()
    canonical = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    body = dict(receipt)
    body["receipt_sha256"] = digest
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body
