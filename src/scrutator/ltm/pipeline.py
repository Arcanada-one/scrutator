"""LTM ingest and recall pipelines — sequential entity/edge extraction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from scrutator.config import settings
from scrutator.db import repository
from scrutator.ltm.models import Entity, EntityEdge, EntityEvent, RecallResult
from scrutator.ltm.prompts import format_dedup, format_edge_extraction, format_entity_extraction, format_rerank
from scrutator.ltm.temporal import DateExtractor

if TYPE_CHECKING:
    from scrutator.ltm.llm import LtmLlmClient

log = logging.getLogger("scrutator.ltm.pipeline")


def timezone_utc() -> type[UTC.__class__] | type[UTC]:  # noqa: ANN401
    """Indirection — lets tests monkey-patch the 'now' timezone."""
    return UTC


def _temporal_score(events: list[dict], ref: datetime) -> float:
    """Recency score in [0, 1]; older → smaller. Chunks without events → neutral 0.5."""
    if not events:
        return 0.5
    timestamps: list[datetime] = []
    for ev in events:
        ts = ev.get("when_t") or ev.get("valid_from")
        if isinstance(ts, datetime):
            timestamps.append(ts if ts.tzinfo else ts.replace(tzinfo=UTC))
    if not timestamps:
        return 0.5
    most_recent = max(timestamps)
    age_days = max(0.0, (ref - most_recent).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / 365.0)


class IngestPipeline:
    """Sequential ingest: for each chunk, extract entities then edges."""

    def __init__(
        self,
        llm: LtmLlmClient,
        namespace: str,
        namespace_id: int,
        max_entities_per_chunk: int = 10,
    ):
        self.llm = llm
        self.namespace = namespace
        self.namespace_id = namespace_id
        self.max_entities_per_chunk = max_entities_per_chunk

    async def extract_entities(self, content: str) -> list[Entity]:
        """Extract entities from a text chunk via LLM."""
        system, user = format_entity_extraction(content, self.max_entities_per_chunk)
        raw = await self.llm.extract_json(user, system=system)

        if not isinstance(raw, list):
            log.warning("Entity extraction returned non-list: %s", type(raw).__name__)
            return []

        entities: list[Entity] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            etype = item.get("type", "").strip()
            if not name or not etype:
                continue
            entities.append(
                Entity(
                    name=name,
                    entity_type=etype,
                    description=item.get("description"),
                )
            )
            if len(entities) >= self.max_entities_per_chunk:
                break

        return entities

    async def extract_edges(self, content: str, entities: list[Entity]) -> list[EntityEdge]:
        """Extract edges between entities via LLM."""
        if len(entities) < 2:
            return []

        entity_dicts = [{"name": e.name, "type": e.entity_type} for e in entities]
        system, user = format_edge_extraction(content, entity_dicts)
        raw = await self.llm.extract_json(user, system=system)

        if not isinstance(raw, list):
            log.warning("Edge extraction returned non-list: %s", type(raw).__name__)
            return []

        entity_names = {e.name for e in entities}
        edges: list[EntityEdge] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "").strip()
            target = item.get("target", "").strip()
            relation = item.get("relation", "").strip()
            if not source or not target or not relation:
                continue
            if source not in entity_names or target not in entity_names:
                continue
            edges.append(EntityEdge(source=source, target=target, relation=relation))

        return edges

    async def dedup_entities(self, entity_names: list[str]) -> list[dict]:
        """Ask LLM to group duplicate entity names. Returns list of {canonical, aliases}."""
        if len(entity_names) < 2:
            return []

        system, user = format_dedup(entity_names)
        raw = await self.llm.extract_json(user, system=system)

        if not isinstance(raw, list):
            log.warning("Dedup returned non-list: %s", type(raw).__name__)
            return []

        groups: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            canonical = item.get("canonical", "").strip()
            aliases = item.get("aliases", [])
            if not canonical:
                continue
            if not isinstance(aliases, list):
                continue
            groups.append({"canonical": canonical, "aliases": [a for a in aliases if isinstance(a, str)]})

        return groups

    async def process_chunk(self, chunk_id: str, content: str) -> tuple[list[Entity], list[EntityEdge]]:
        """Process a single chunk: extract entities + edges, persist to DB."""
        entities = await self.extract_entities(content)
        edges = await self.extract_edges(content, entities)

        # Persist entities
        entity_id_map: dict[str, str] = {}
        for entity in entities:
            eid = await repository.upsert_entity(
                namespace_id=self.namespace_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                properties=entity.properties,
                source_chunk_id=chunk_id,
            )
            entity_id_map[entity.name] = eid

        # Persist entity edges
        for edge in edges:
            src_id = entity_id_map.get(edge.source)
            tgt_id = entity_id_map.get(edge.target)
            if src_id and tgt_id:
                await repository.upsert_entity_edge(
                    source_entity_id=src_id,
                    target_entity_id=tgt_id,
                    relation=edge.relation,
                    weight=edge.weight,
                    source_chunk_id=chunk_id,
                )

        # LTM-0012 — temporal events (gated by config flag)
        if settings.ltm_temporal_enabled and entity_id_map:
            await self._extract_and_persist_events(
                chunk_id=chunk_id,
                content=content,
                entity_id_map=entity_id_map,
            )

        return entities, edges

    async def _extract_and_persist_events(
        self,
        chunk_id: str,
        content: str,
        entity_id_map: dict[str, str],
    ) -> list[EntityEvent]:
        """Extract temporal events for known entities, persist with auto-invalidate."""
        extractor = DateExtractor(llm=self.llm, max_events=settings.ltm_max_events_per_chunk)
        try:
            events = await extractor.extract(content, list(entity_id_map.keys()))
        except Exception:
            log.exception("temporal extraction failed for chunk %s", chunk_id)
            return []

        for ev in events:
            entity_id = entity_id_map.get(ev.entity_name)
            if not entity_id:
                continue
            try:
                new_id = await repository.upsert_entity_event(
                    namespace_id=self.namespace_id,
                    entity_id=entity_id,
                    event_type=ev.event_type,
                    when_t=ev.when_t,
                    valid_from=ev.valid_from,
                    valid_to=ev.valid_to,
                    description=ev.description,
                    properties=ev.properties,
                    source_chunk_id=chunk_id,
                )
            except Exception:
                log.exception("upsert_entity_event failed for %s", ev.entity_name)
                continue

            if settings.ltm_auto_invalidate and ev.valid_from is not None:
                await self._supersede_overlaps(
                    entity_id=entity_id,
                    event_type=ev.event_type,
                    new_valid_from=ev.valid_from,
                    new_event_id=new_id,
                )
        return events

    async def _supersede_overlaps(
        self,
        entity_id: str,
        event_type: str,
        new_valid_from: datetime,
        new_event_id: str,
    ) -> None:
        """Close prior open events that overlap with the new event."""
        overlaps = await repository.find_overlapping_events(
            namespace_id=self.namespace_id,
            entity_id=entity_id,
            event_type=event_type,
            valid_from=new_valid_from,
            exclude_id=new_event_id,
        )
        if not overlaps:
            return
        delta = timedelta(microseconds=1)
        cutoff = new_valid_from - delta
        for old in overlaps:
            await repository.supersede_event(
                event_id=old["id"],
                valid_to=cutoff,
                superseded_by=new_event_id,
            )


class RecallPipeline:
    """Entity-enriched recall: search + expand entities."""

    def __init__(self, llm: LtmLlmClient, namespace: str, namespace_id: int):
        self.llm = llm
        self.namespace = namespace
        self.namespace_id = namespace_id

    async def filter_temporal(
        self,
        results: list[dict],
        as_of: datetime | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[dict]:
        """LTM-0012 — drop chunks whose events don't match the time filter.
        Chunks without any event are kept (treated as timeless)."""
        if as_of is None and time_range is None:
            return results
        if not results:
            return results
        chunk_ids = [r["chunk_id"] for r in results]
        kept_ids = set(
            await repository.filter_chunks_by_temporal(
                chunk_ids=chunk_ids,
                as_of=as_of,
                time_range=time_range,
            )
        )
        return [r for r in results if r["chunk_id"] in kept_ids]

    def apply_temporal_boost(
        self,
        results: list[RecallResult],
        events_by_chunk: dict[str, list[dict]],
        boost: float,
        now: datetime | None = None,
    ) -> list[RecallResult]:
        """Recency-weighted boost: score' = score + boost * (1 / (1 + age_years)).
        Chunks without events get neutral 0.5."""
        if boost <= 0.0 or not results:
            return results
        ref = now or datetime.now(tz=timezone_utc())
        boosted: list[tuple[float, RecallResult]] = []
        for r in results:
            events = events_by_chunk.get(r.chunk_id, [])
            t_score = _temporal_score(events, ref)
            boosted.append((r.score + boost * t_score, r))
        boosted.sort(key=lambda pair: pair[0], reverse=True)
        return [r.model_copy(update={"score": new_score}) for new_score, r in boosted]

    async def rerank(self, query: str, results: list[RecallResult]) -> list[RecallResult]:
        """Rerank results via LLM. Falls back to original order on failure."""
        if len(results) < 2:
            return results

        candidates = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "entities": [{"name": e.name} for e in r.entities],
            }
            for r in results
        ]
        system, user = format_rerank(query, candidates)
        raw = await self.llm.extract_json(user, system=system)

        if not isinstance(raw, list):
            log.warning("Rerank returned non-list, preserving original order")
            return results

        # Build lookup and reorder
        result_map = {r.chunk_id: r for r in results}
        reranked: list[RecallResult] = []
        seen: set[str] = set()
        for cid in raw:
            if not isinstance(cid, str):
                continue
            if cid in result_map and cid not in seen:
                reranked.append(result_map[cid])
                seen.add(cid)

        # Append any results not mentioned by LLM (preserve their relative order)
        for r in results:
            if r.chunk_id not in seen:
                reranked.append(r)

        return reranked

    async def enrich_with_entities(self, search_results: list[dict]) -> list[RecallResult]:
        """Enrich search results with entity and edge context."""
        if not search_results:
            return []

        chunk_ids = [r["chunk_id"] for r in search_results]
        entities_map = await repository.get_entities_for_chunks(chunk_ids)
        edges_map = await repository.get_entity_edges_for_chunks(chunk_ids)

        enriched: list[RecallResult] = []
        for r in search_results:
            cid = r["chunk_id"]
            chunk_entities = [
                Entity(
                    name=e["name"],
                    entity_type=e["entity_type"],
                    description=e.get("description"),
                    properties=e.get("properties", {}),
                )
                for e in entities_map.get(cid, [])
            ]
            chunk_edges = [
                EntityEdge(
                    source=e["source_name"],
                    target=e["target_name"],
                    relation=e["relation"],
                    weight=e.get("weight", 1.0),
                )
                for e in edges_map.get(cid, [])
            ]
            enriched.append(
                RecallResult(
                    chunk_id=cid,
                    content=r.get("content", ""),
                    source_path=r.get("source_path", ""),
                    score=r.get("score", 0.0),
                    namespace=r.get("namespace", self.namespace),
                    project=r.get("project"),
                    metadata=r.get("metadata", {}),
                    entities=chunk_entities,
                    relations=chunk_edges,
                )
            )

        return enriched
