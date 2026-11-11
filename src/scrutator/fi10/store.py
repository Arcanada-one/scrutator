"""In-memory synthetic FI10 generation store (vector + typed-graph projections).

Labels every fixture synthetic. No network, no production Scrutator writes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from scrutator.fi10.namespaces import FI10_PRIVATE, FI10_PUBLIC, DisclosureClass

Projection = Literal["vector", "typed_graph"]


@dataclass(frozen=True)
class Fi10Record:
    record_id: str
    namespace: str
    disclosure_class: DisclosureClass
    generation_id: str
    body: str
    synthetic: Literal[True] = True

    def content_digest(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Fi10Relation:
    edge_id: str
    namespace: str
    generation_id: str
    source_record_id: str
    target_record_id: str
    relation: str
    synthetic: Literal[True] = True


@dataclass
class SyntheticFi10Store:
    """Disposable store for FI10-T5.2 readback and purge proofs."""

    database: str = "synthetic-fi10"
    collection: str = "scrutator-fi10"
    records: dict[str, Fi10Record] = field(default_factory=dict)
    relations: dict[str, Fi10Relation] = field(default_factory=dict)
    purged_record_ids: set[str] = field(default_factory=set)
    revoked_generations: set[str] = field(default_factory=set)
    last_known_good_generation: str | None = None
    active_generation: str | None = None

    def upsert_record(self, record: Fi10Record) -> None:
        if record.namespace not in {FI10_PUBLIC, FI10_PRIVATE}:
            raise ValueError(f"unsupported namespace {record.namespace!r}")
        if not record.synthetic:
            raise ValueError("only synthetic records are accepted")
        self.records[record.record_id] = record
        self.purged_record_ids.discard(record.record_id)
        self.active_generation = record.generation_id
        if self.last_known_good_generation is None:
            self.last_known_good_generation = record.generation_id

    def upsert_relation(self, relation: Fi10Relation) -> None:
        if relation.namespace not in {FI10_PUBLIC, FI10_PRIVATE}:
            raise ValueError(f"unsupported namespace {relation.namespace!r}")
        self.relations[relation.edge_id] = relation

    def mark_revoked(self, generation_id: str) -> None:
        self.revoked_generations.add(generation_id)

    def visible_records(
        self,
        *,
        namespace: str,
        generation_id: str | None = None,
    ) -> list[Fi10Record]:
        out: list[Fi10Record] = []
        for record in self.records.values():
            if record.record_id in self.purged_record_ids:
                continue
            if record.namespace != namespace:
                continue
            if generation_id is not None and record.generation_id != generation_id:
                continue
            out.append(record)
        return sorted(out, key=lambda r: r.record_id)

    def visible_relations(
        self,
        *,
        namespace: str,
        generation_id: str | None = None,
    ) -> list[Fi10Relation]:
        live_ids = {r.record_id for r in self.visible_records(namespace=namespace, generation_id=generation_id)}
        out: list[Fi10Relation] = []
        for edge in self.relations.values():
            if edge.namespace != namespace:
                continue
            if generation_id is not None and edge.generation_id != generation_id:
                continue
            if edge.source_record_id not in live_ids or edge.target_record_id not in live_ids:
                continue
            out.append(edge)
        return sorted(out, key=lambda e: e.edge_id)

    def to_manifest_slice(self, generation_id: str) -> dict[str, Any]:
        records = [r for r in self.records.values() if r.generation_id == generation_id]
        return {
            "schema": "SyntheticFi10GenerationManifest/v1",
            "label": "synthetic",
            "generation_id": generation_id,
            "record_digests": sorted(r.content_digest() for r in records),
            "record_ids": sorted(r.record_id for r in records),
        }

    def dump_json(self) -> str:
        payload = {
            "database": self.database,
            "collection": self.collection,
            "active_generation": self.active_generation,
            "last_known_good_generation": self.last_known_good_generation,
            "revoked_generations": sorted(self.revoked_generations),
            "purged_record_ids": sorted(self.purged_record_ids),
            "records": [
                {
                    "record_id": r.record_id,
                    "namespace": r.namespace,
                    "disclosure_class": r.disclosure_class,
                    "generation_id": r.generation_id,
                    "body": r.body,
                    "synthetic": True,
                    "content_digest": r.content_digest(),
                }
                for r in sorted(self.records.values(), key=lambda x: x.record_id)
            ],
            "relations": [
                {
                    "edge_id": e.edge_id,
                    "namespace": e.namespace,
                    "generation_id": e.generation_id,
                    "source_record_id": e.source_record_id,
                    "target_record_id": e.target_record_id,
                    "relation": e.relation,
                    "synthetic": True,
                }
                for e in sorted(self.relations.values(), key=lambda x: x.edge_id)
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
