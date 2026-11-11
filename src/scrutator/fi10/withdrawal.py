"""Withdrawal purge and last-known-good / forbidden-fallback rules for FI10."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from scrutator.fi10.store import SyntheticFi10Store


@dataclass(frozen=True)
class PurgeResult:
    schema: str
    label: str
    record_id: str
    generation_id: str
    purged: bool
    search_entries_remaining: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServingDecision:
    verdict: Literal["serve", "refuse"]
    generation_id: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Fi10WithdrawalError(Exception):
    """Named refusal for withdrawal / fallback decisions."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def purge_withdrawn(store: SyntheticFi10Store, *, record_id: str, generation_id: str) -> PurgeResult:
    """Remove search entries for a withdrawn record and prove absence on readback path."""
    record = store.records.get(record_id)
    if record is None:
        return PurgeResult(
            schema="Fi10WithdrawalPurge/v1",
            label="synthetic",
            record_id=record_id,
            generation_id=generation_id,
            purged=False,
            search_entries_remaining=0,
            reason="record_absent",
        )
    if record.generation_id != generation_id:
        raise Fi10WithdrawalError(
            "GENERATION_MISMATCH",
            f"record {record_id} belongs to {record.generation_id}, not {generation_id}",
        )
    store.purged_record_ids.add(record_id)
    remaining = len(
        [
            r
            for r in store.visible_records(namespace=record.namespace, generation_id=generation_id)
            if r.record_id == record_id
        ]
    )
    return PurgeResult(
        schema="Fi10WithdrawalPurge/v1",
        label="synthetic",
        record_id=record_id,
        generation_id=generation_id,
        purged=True,
        search_entries_remaining=remaining,
        reason="purged",
    )


def resolve_serving_generation(
    store: SyntheticFi10Store,
    *,
    requested_generation: str | None,
    projection_failed: bool = False,
) -> ServingDecision:
    """Serve the requested generation, or LKG on projection failure — except revoked.

    A revoked disclosure generation must never fall back.
    """
    if requested_generation is not None and requested_generation in store.revoked_generations:
        return ServingDecision("refuse", None, "revoked_generation_forbidden_fallback")

    if projection_failed:
        lkg = store.last_known_good_generation
        if lkg is None:
            return ServingDecision("refuse", None, "no_last_known_good")
        if lkg in store.revoked_generations:
            return ServingDecision("refuse", None, "revoked_generation_forbidden_fallback")
        return ServingDecision("serve", lkg, "last_known_good_on_projection_failure")

    if requested_generation is None:
        active = store.active_generation
        if active is None:
            return ServingDecision("refuse", None, "no_active_generation")
        if active in store.revoked_generations:
            return ServingDecision("refuse", None, "revoked_generation_forbidden_fallback")
        return ServingDecision("serve", active, "active_generation")

    if requested_generation in store.revoked_generations:
        return ServingDecision("refuse", None, "revoked_generation_forbidden_fallback")
    return ServingDecision("serve", requested_generation, "requested_generation")
