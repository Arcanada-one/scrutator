"""LTM ingest and recall pipelines — sequential entity/edge extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scrutator.db import repository
from scrutator.ltm.models import Entity, EntityEdge, RecallResult
from scrutator.ltm.prompts import format_dedup, format_edge_extraction, format_entity_extraction, format_rerank

if TYPE_CHECKING:
    from scrutator.ltm.llm import LtmLlmClient

log = logging.getLogger("scrutator.ltm.pipeline")


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

        return entities, edges


class RecallPipeline:
    """Entity-enriched recall: search + expand entities."""

    def __init__(self, llm: LtmLlmClient, namespace: str, namespace_id: int):
        self.llm = llm
        self.namespace = namespace
        self.namespace_id = namespace_id

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
