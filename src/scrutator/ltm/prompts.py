"""Prompt templates for LTM entity/edge extraction."""

_ENTITY_EXTRACTION_SYSTEM = (
    "You are a knowledge graph entity extractor.\n"
    "Extract named entities from the given text.\n"
    "Return ONLY a JSON array, no other text.\n"
    'Each entity: {{"name": "...", "type": "...", "description": "..."}}\n'
    "Valid types: person, project, concept, technology, event, organization, location.\n"
    "Maximum {max_entities} entities per chunk."
)

ENTITY_EXTRACTION_USER = """Extract entities from this text:

{content}"""

EDGE_EXTRACTION_SYSTEM = """You are a knowledge graph relationship extractor.
Given entities and their source text, identify relationships between them.
Return ONLY a JSON array, no other text.
Each relationship: {"source": "entity_name", "target": "entity_name", "relation": "verb_phrase"}
Use short relation names: works_on, depends_on, part_of, created_by, located_in, related_to, etc."""

EDGE_EXTRACTION_USER = """Entities found:
{entity_list}

Source text:
{content}"""

DEDUP_SYSTEM = """You are a deduplication assistant.
Group entity names that refer to the same real-world concept.
Return ONLY a JSON array, no other text.
Each group: {"canonical": "best_name", "aliases": ["alt1", "alt2"]}
Only group true duplicates (same entity, different spelling/casing)."""

DEDUP_USER = """These entity names may be duplicates. Group them:
{entity_names}"""

RERANK_SYSTEM = """You are a search result reranker.
Given a query and candidate search results with entity context, reorder by relevance.
Return ONLY a JSON array of chunk_ids in order of relevance, most relevant first.
Example: ["chunk-id-1", "chunk-id-2", "chunk-id-3"]"""

RERANK_USER = """Query: {query}

Candidates:
{candidates}"""


def format_entity_extraction(content: str, max_entities: int = 10) -> tuple[str, str]:
    """Return (system, user) prompts for entity extraction."""
    system = _ENTITY_EXTRACTION_SYSTEM.format(max_entities=max_entities)
    user = ENTITY_EXTRACTION_USER.format(content=content[:4000])
    return system, user


def format_edge_extraction(content: str, entities: list[dict]) -> tuple[str, str]:
    """Return (system, user) prompts for edge extraction."""
    entity_list = "\n".join(f"- {e['name']} ({e.get('type', 'unknown')})" for e in entities)
    user = EDGE_EXTRACTION_USER.format(entity_list=entity_list, content=content[:4000])
    return EDGE_EXTRACTION_SYSTEM, user


def format_dedup(entity_names: list[str]) -> tuple[str, str]:
    """Return (system, user) prompts for entity dedup."""
    names = "\n".join(f"- {n}" for n in entity_names)
    user = DEDUP_USER.format(entity_names=names)
    return DEDUP_SYSTEM, user


def format_rerank(query: str, candidates: list[dict]) -> tuple[str, str]:
    """Return (system, user) prompts for reranking."""
    lines = []
    for c in candidates:
        entities_str = ", ".join(e.get("name", "") for e in c.get("entities", []))
        lines.append(f"ID: {c['chunk_id']}\nContent: {c['content'][:200]}\nEntities: {entities_str}\n")
    user = RERANK_USER.format(query=query, candidates="\n".join(lines))
    return RERANK_SYSTEM, user


# LTM-0012 — temporal event extraction (LLM Layer 2 fallback)
EVENT_EXTRACTION_SYSTEM = (
    "You are a temporal fact extractor.\n"
    "Given a text chunk and a list of known entities, return events with timestamps.\n"
    "Return ONLY a JSON array, no other text. Empty array [] when there are no events.\n"
    "Each event:\n"
    '  {"entity_name": "<must match a known entity>",\n'
    '   "event_type": "<short>",\n'
    '   "when": "ISO-8601 date or null",\n'
    '   "valid_from": "ISO-8601 date or null",\n'
    '   "valid_to": "ISO-8601 date or null",\n'
    '   "description": "<10-word summary>"}\n'
    "Allowed event_type values: archived, created, completed, started, released, "
    "deployed, updated, deprecated, superseded.\n"
    "Use ISO-8601 date format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ). "
    "Use null when unknown — do not invent dates."
)

EVENT_EXTRACTION_USER = """Known entities:
{entity_list}

Text:
\"\"\"{content}\"\"\""""


def format_event_extraction(content: str, entity_names: list[str]) -> tuple[str, str]:
    """Return (system, user) prompts for temporal event extraction (LTM-0012)."""
    entity_list = "\n".join(f"- {n}" for n in entity_names) if entity_names else "- (none)"
    user = EVENT_EXTRACTION_USER.format(entity_list=entity_list, content=content[:4000])
    return EVENT_EXTRACTION_SYSTEM, user
