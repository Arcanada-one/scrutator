"""Generation readback for FI10 namespaces (vector + typed-graph where claimed)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from scrutator.fi10.namespaces import FI10_PRIVATE, FI10_PUBLIC, decide_read
from scrutator.fi10.store import SyntheticFi10Store


@dataclass(frozen=True)
class NamespaceCounts:
    namespace: str
    record_count: int
    relation_count: int
    vector_hits: int
    typed_graph_hits: int


@dataclass(frozen=True)
class GenerationReadback:
    schema: str
    label: str
    database: str
    collection: str
    generation_id: str
    namespaces: tuple[NamespaceCounts, ...]
    positive_control_record_id: str | None
    positive_control_found: bool
    private_records_in_public: int
    runtime_authorized: bool
    knowledge_admitted: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["namespaces"] = [asdict(n) for n in self.namespaces]
        return payload


def scoped_query(
    store: SyntheticFi10Store,
    *,
    principal_id: str,
    granted_namespaces: frozenset[str],
    namespace: str,
    generation_id: str,
) -> list[str]:
    """Return record ids visible to the principal in one namespace at one generation."""
    visible: list[str] = []
    for record in store.visible_records(namespace=namespace, generation_id=generation_id):
        decision = decide_read(
            principal_id=principal_id,
            granted_namespaces=granted_namespaces,
            namespace=namespace,
            disclosure_class=record.disclosure_class,
        )
        if decision.verdict == "allow":
            visible.append(record.record_id)
    return visible


def readback_generation(
    store: SyntheticFi10Store,
    *,
    generation_id: str,
    principal_id: str,
    granted_namespaces: frozenset[str],
    positive_control_record_id: str,
) -> GenerationReadback:
    """Name database/collection, namespace, record/relation counts and exact generation.

    Positive visibility control: the named public record must be found in fi10-public
    for the same generation when the principal holds that grant.
    """
    public_ids = scoped_query(
        store,
        principal_id=principal_id,
        granted_namespaces=granted_namespaces,
        namespace=FI10_PUBLIC,
        generation_id=generation_id,
    )
    private_ids = scoped_query(
        store,
        principal_id=principal_id,
        granted_namespaces=granted_namespaces,
        namespace=FI10_PRIVATE,
        generation_id=generation_id,
    )

    # Negative control seam: count private-classified bodies that would leak into public.
    private_in_public = 0
    for record in store.visible_records(namespace=FI10_PUBLIC, generation_id=generation_id):
        if record.disclosure_class == "private":
            private_in_public += 1

    public_relations = store.visible_relations(namespace=FI10_PUBLIC, generation_id=generation_id)
    private_relations = store.visible_relations(namespace=FI10_PRIVATE, generation_id=generation_id)

    namespaces = (
        NamespaceCounts(
            namespace=FI10_PUBLIC,
            record_count=len(public_ids),
            relation_count=len(public_relations),
            vector_hits=len(public_ids),
            typed_graph_hits=len(public_relations),
        ),
        NamespaceCounts(
            namespace=FI10_PRIVATE,
            record_count=len(private_ids),
            relation_count=len(private_relations),
            vector_hits=len(private_ids),
            typed_graph_hits=len(private_relations),
        ),
    )

    positive_found = positive_control_record_id in public_ids
    return GenerationReadback(
        schema="Fi10GenerationReadback/v1",
        label="synthetic",
        database=store.database,
        collection=store.collection,
        generation_id=generation_id,
        namespaces=namespaces,
        positive_control_record_id=positive_control_record_id,
        positive_control_found=positive_found,
        private_records_in_public=private_in_public,
        runtime_authorized=False,
        knowledge_admitted=False,
    )
