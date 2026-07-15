"""Canonical Muneral aggregate and deterministic Scrutator graph mapping."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

_VOLATILE_KEYS = {"captured_at", "run_id"}
_TASK_FIELDS = (
    "id",
    "project_id",
    "sprint_id",
    "parent_id",
    "title",
    "description",
    "status",
    "priority",
    "due_date",
    "estimate_hours",
    "created_by_id",
    "actor_type",
    "created_at",
    "updated_at",
)
_DEPENDENCY_RELATIONS = {
    "depends_on": "depends-on",
    "blocks": "blocks",
    "related_to": "related-to",
    "duplicates": "duplicate-of",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items() if key not in _VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    return value


def normalize_token(value: str) -> str:
    """Return a stable lowercase relation/tag token."""
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[\W_]+", "-", normalized, flags=re.UNICODE).strip("-")


def _normalize_dependencies(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = [
        {
            "type": str(item["type"]),
            "from_task_id": str(item["from_task_id"]),
            "to_task_id": str(item["to_task_id"]),
        }
        for item in items
    ]
    return sorted(normalized, key=lambda item: (item["type"], item["from_task_id"], item["to_task_id"]))


def _normalize_checklists(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {
            "id": str(item["id"]),
            "text": str(item["text"]),
            "checked": bool(item["checked"]),
            "position": int(item["position"]) if item.get("position") is not None else None,
        }
        for item in items
    ]
    return sorted(
        normalized,
        key=lambda item: (
            item["position"] is None,
            item["position"] if item["position"] is not None else 0,
            item["id"],
        ),
    )


def _normalize_agents(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = [{"agent_id": str(item["agent_id"]), "role": normalize_token(str(item["role"]))} for item in items]
    return sorted(normalized, key=lambda item: (item["agent_id"], item["role"]))


def _normalize_activity(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {
            "id": str(item["id"]),
            "actor_type": str(item["actor_type"]),
            "actor_id": str(item["actor_id"]) if item.get("actor_id") is not None else None,
            "action": str(item["action"]),
            "created_at": _json_value(item["created_at"]),
        }
        for item in items
    ]
    return sorted(normalized, key=lambda item: (str(item["created_at"]), item["id"]))


def canonical_snapshot(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Select, normalize, and deterministically order source-controlled fields."""
    task = {field: _json_value(aggregate["task"].get(field)) for field in _TASK_FIELDS}
    project_source = aggregate.get("project") or {}
    project = {
        "id": _json_value(project_source.get("id")),
        "name": _json_value(project_source.get("name")),
        "slug": _json_value(project_source.get("slug")),
    }
    tags = sorted({normalize_token(str(tag)) for tag in aggregate.get("tags", []) if normalize_token(str(tag))})
    return {
        "task": task,
        "project": project,
        "tags": tags,
        "dependencies": _normalize_dependencies(aggregate.get("dependencies", [])),
        "checklists": _normalize_checklists(aggregate.get("checklists", [])),
        "agents": _normalize_agents(aggregate.get("agents", [])),
        "activity": _normalize_activity(aggregate.get("activity", [])),
    }


def canonical_bytes(aggregate: dict[str, Any]) -> bytes:
    return json.dumps(canonical_snapshot(aggregate), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def canonical_hash(aggregate: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(aggregate)).hexdigest()


def _entity(
    name: str,
    entity_type: str,
    source_ref: str,
    content_hash: str,
    *,
    description: str | None = None,
    **properties: Any,
) -> dict[str, Any]:
    entity = {
        "name": name,
        "entity_type": entity_type,
        "properties": {"source_ref": source_ref, "content_hash": content_hash, **properties},
    }
    if description is not None:
        entity["description"] = description
    return entity


def _identity_entity(
    name: str,
    entity_type: str,
    source_ref: str,
    *,
    description: str | None = None,
    **properties: Any,
) -> dict[str, Any]:
    identity = {"entity_type": entity_type, "source_ref": source_ref, **properties}
    identity_hash = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _entity(
        name,
        entity_type,
        source_ref,
        identity_hash,
        description=description,
        **properties,
    )


def _task_stub(task_id: str) -> dict[str, Any]:
    return {"name": f"MUN:{task_id}", "entity_type": "task", "properties": {"muneral_id": task_id}}


def _render_content(snapshot: dict[str, Any]) -> str:
    task = snapshot["task"]
    project = snapshot["project"]
    lines = [
        f"Muneral task: {task['title']}",
        f"Task ID: {task['id']}",
        f"Status: {task['status']}",
        f"Priority: {task['priority']}",
        f"Project: {project['name']} ({project['id']})",
        f"Tags: {', '.join(snapshot['tags'])}",
        f"Description: {task['description'] or ''}",
    ]
    if task["parent_id"]:
        lines.append(f"Parent task: {task['parent_id']}")
    if snapshot["checklists"]:
        lines.append(
            "Checklist: "
            + "; ".join(f"[{'x' if item['checked'] else ' '}] {item['text']}" for item in snapshot["checklists"])
        )
    if snapshot["activity"]:
        lines.append("Activity: " + "; ".join(item["action"] for item in snapshot["activity"]))
    return "\n".join(lines)


def _task_properties(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "muneral_id": str(task["id"]),
        "display_name": task["title"],
        "status": task["status"],
        "priority": task["priority"],
        "actor_type": task["actor_type"],
        "created_by_id": task["created_by_id"],
        "project_id": task["project_id"],
        "sprint_id": task["sprint_id"],
        "parent_id": task["parent_id"],
        "due_date": task["due_date"],
        "estimate_hours": task["estimate_hours"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
    }


def _add_project_and_parent(
    snapshot: dict[str, Any],
    source_ref: str,
    content_hash: str,
    entities: dict[str, dict[str, Any]],
    edges: set[tuple[str, str, str]],
) -> None:
    task = snapshot["task"]
    task_name = f"MUN:{task['id']}"
    project = snapshot["project"]
    if project["id"]:
        project_name = f"MUN-PROJECT:{project['id']}"
        entities[project_name] = _identity_entity(
            project_name,
            "project",
            f"muneral://project/{project['id']}",
            description=str(project["name"]),
            muneral_id=project["id"],
            display_name=project["name"],
        )
        edges.add((task_name, project_name, "belongs-to-project"))

    if task["parent_id"]:
        parent_name = f"MUN:{task['parent_id']}"
        entities[parent_name] = _task_stub(str(task["parent_id"]))
        edges.add((task_name, parent_name, "subtask-of"))


def _add_tags_and_actors(
    snapshot: dict[str, Any],
    source_ref: str,
    content_hash: str,
    entities: dict[str, dict[str, Any]],
    edges: set[tuple[str, str, str]],
) -> None:
    task = snapshot["task"]
    task_name = f"MUN:{task['id']}"
    for tag in snapshot["tags"]:
        tag_name = f"MUN-TAG:{tag}"
        entities[tag_name] = _identity_entity(
            tag_name,
            "tag",
            f"muneral://tag/{tag}",
            description=tag,
            muneral_id=tag,
            display_name=tag,
        )
        edges.add((task_name, tag_name, "tagged"))

    if task["created_by_id"]:
        actor_type = normalize_token(str(task["actor_type"] or "unknown"))
        actor_name = f"MUN-ACTOR:{actor_type}:{task['created_by_id']}"
        entities[actor_name] = _identity_entity(
            actor_name,
            "actor",
            f"muneral://actor/{actor_type}/{task['created_by_id']}",
            muneral_id=task["created_by_id"],
            actor_type=actor_type,
        )
        edges.add((task_name, actor_name, "performed-by"))

    for assignment in snapshot["agents"]:
        actor_name = f"MUN-ACTOR:agent:{assignment['agent_id']}"
        entities[actor_name] = _identity_entity(
            actor_name,
            "actor",
            f"muneral://actor/agent/{assignment['agent_id']}",
            muneral_id=assignment["agent_id"],
            actor_type="agent",
        )
        edges.add((task_name, actor_name, f"assigned-{assignment['role']}"))


def _add_dependencies(
    snapshot: dict[str, Any],
    source_ref: str,
    content_hash: str,
    entities: dict[str, dict[str, Any]],
    edges: set[tuple[str, str, str]],
) -> None:
    for dependency in snapshot["dependencies"]:
        source = f"MUN:{dependency['from_task_id']}"
        target = f"MUN:{dependency['to_task_id']}"
        entities.setdefault(source, _task_stub(str(dependency["from_task_id"])))
        entities.setdefault(target, _task_stub(str(dependency["to_task_id"])))
        relation = _DEPENDENCY_RELATIONS.get(dependency["type"])
        if relation is None:
            raise ValueError(f"unsupported dependency type: {dependency['type']}")
        edges.add((source, target, relation))


def build_ingest_payload(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Build the schema-v1 structured graph request for one task aggregate."""
    snapshot = canonical_snapshot(aggregate)
    task = snapshot["task"]
    project = snapshot["project"]
    task_id = str(task["id"])
    source_ref = f"muneral://task/{task_id}"
    content_hash = canonical_hash(aggregate)
    task_name = f"MUN:{task_id}"
    entities = {
        task_name: _entity(
            task_name,
            "task",
            source_ref,
            content_hash,
            description=str(task["title"]),
            **_task_properties(task),
        )
    }
    edges: set[tuple[str, str, str]] = set()
    _add_project_and_parent(snapshot, source_ref, content_hash, entities, edges)
    _add_tags_and_actors(snapshot, source_ref, content_hash, entities, edges)
    _add_dependencies(snapshot, source_ref, content_hash, entities, edges)
    graph_entities = sorted(entities.values(), key=lambda item: (item["name"], item["entity_type"]))
    graph_edges = [
        {"source": source, "target": target, "relation": relation, "weight": 1.0}
        for source, target, relation in sorted(edges)
    ]
    return {
        "content": _render_content(snapshot),
        "source_path": source_ref,
        "namespace": "muneral",
        "project": str(project["id"]) if project["id"] else None,
        "structured_graph": {
            "schema_version": 1,
            "content_hash": content_hash,
            "entities": graph_entities,
            "edges": graph_edges,
        },
    }
