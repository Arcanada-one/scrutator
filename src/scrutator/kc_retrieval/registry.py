"""LocationRegistry/v1 loader and role-ordered copy resolution (KC-PIPE-S2 rematch)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

SOURCE_ID = re.compile(r"^d_[0-9a-f]{64}$")
PACKAGE_ID = re.compile(r"^d_[0-9a-f]{64}/[A-Za-z0-9][A-Za-z0-9._@+-]{0,119}$")
ROLE_ORDER: tuple[Literal["primary", "mirror", "backup"], ...] = ("primary", "mirror", "backup")


class RegistryError(Exception):
    """Raised when the configured registry document is unusable."""


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegistryError(f"registry file missing: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"registry unreadable: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != "LocationRegistry/v1":
        raise RegistryError("registry schema must be LocationRegistry/v1")
    entries = doc.get("entries")
    if not isinstance(entries, list):
        raise RegistryError("registry entries must be a list")
    return doc


def object_kind_for(object_id: str) -> Literal["original", "package"]:
    if SOURCE_ID.fullmatch(object_id):
        return "original"
    if PACKAGE_ID.fullmatch(object_id):
        return "package"
    raise ValueError("invalid object id")


def rows_for_object(doc: dict[str, Any], object_id: str) -> list[dict[str, Any]]:
    kind = object_kind_for(object_id)
    matched: list[dict[str, Any]] = []
    for row in doc.get("entries", []):
        if not isinstance(row, dict):
            continue
        if row.get("schema") != "LocationRegistryRow/v1":
            continue
        if row.get("object_id") != object_id:
            continue
        if row.get("object_kind") != kind:
            continue
        if row.get("role") not in ROLE_ORDER:
            continue
        matched.append(row)
    return matched


def ordered_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer primary, then mirror, then backup (LOC-R006 / LOC-R007)."""
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    for row in rows:
        by_role[row["role"]].append(row)
    ordered: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        ordered.extend(by_role[role])
    return ordered


def expected_sha256(row: dict[str, Any]) -> str | None:
    """Digest that must match on-disk bytes before any content is served."""
    object_id = row["object_id"]
    if row["object_kind"] == "original" and SOURCE_ID.fullmatch(object_id):
        return object_id[2:]
    verified = row.get("verified_sha256")
    if isinstance(verified, str) and re.fullmatch(r"[0-9a-f]{64}", verified):
        return verified
    return None
