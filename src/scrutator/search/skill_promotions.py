"""Immutable, repository-authorized maturity for indexed skill plans."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal

ApprovedMaturity = Literal["validated", "production"]

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ROOT_FIELDS = frozenset({"schema_version", "promotions"})
_ENTRY_FIELDS = frozenset({"source_path", "content_hash", "maturity"})
_APPROVED_MATURITIES = frozenset({"validated", "production"})
_REASONS = frozenset(
    {
        "duplicate_identity",
        "entry_schema",
        "invalid_hash",
        "invalid_json",
        "invalid_maturity",
        "io_error",
        "root_schema",
        "unsafe_path",
    }
)


class SkillPromotionRegistryError(RuntimeError):
    """A bounded, non-echoing failure while loading trusted promotion policy."""

    def __init__(self, reason: str) -> None:
        safe_reason = reason if reason in _REASONS else "root_schema"
        self.reason = safe_reason
        super().__init__(f"invalid skill promotion registry: {safe_reason}")


class _ObjectPairs(list[tuple[str, object]]):
    """Preserve JSON object pairs so duplicate keys cannot be overwritten."""


def _object_pairs(pairs: list[tuple[str, object]]) -> _ObjectPairs:
    return _ObjectPairs(pairs)


def _closed_object(
    value: object,
    fields: frozenset[str],
    reason: Literal["root_schema", "entry_schema"],
) -> dict[str, object]:
    if not isinstance(value, _ObjectPairs):
        raise SkillPromotionRegistryError(reason)
    keys = [key for key, _item in value]
    if any(not isinstance(key, str) for key in keys):
        raise SkillPromotionRegistryError(reason)
    if len(keys) != len(set(keys)) or frozenset(keys) != fields:
        raise SkillPromotionRegistryError(reason)
    return dict(value)


def _is_safe_source_path(source_path: object) -> bool:
    if not isinstance(source_path, str) or source_path in {"", ".", ".."}:
        return False
    if "\\" in source_path or "\x00" in source_path:
        return False
    if unicodedata.normalize("NFC", source_path) != source_path:
        return False
    if any(unicodedata.category(char) in {"Cc", "Cf"} for char in source_path):
        return False
    path = PurePosixPath(source_path)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        return False
    return path.as_posix() == source_path


def load_skill_promotions(raw: str) -> Mapping[tuple[str, str], ApprovedMaturity]:
    """Validate the complete registry and return an immutable exact-pair map."""
    try:
        parsed = json.loads(raw, object_pairs_hook=_object_pairs)
    except (json.JSONDecodeError, RecursionError, UnicodeError, TypeError) as exc:
        raise SkillPromotionRegistryError("invalid_json") from exc

    root = _closed_object(parsed, _ROOT_FIELDS, "root_schema")
    schema_version = root["schema_version"]
    if isinstance(schema_version, bool) or schema_version != 1:
        raise SkillPromotionRegistryError("root_schema")
    promotions = root["promotions"]
    if type(promotions) is not list:
        raise SkillPromotionRegistryError("root_schema")

    loaded: dict[tuple[str, str], ApprovedMaturity] = {}
    for item in promotions:
        entry = _closed_object(item, _ENTRY_FIELDS, "entry_schema")
        source_path = entry["source_path"]
        content_hash = entry["content_hash"]
        maturity = entry["maturity"]
        if not _is_safe_source_path(source_path):
            raise SkillPromotionRegistryError("unsafe_path")
        if not isinstance(content_hash, str) or _HASH_RE.fullmatch(content_hash) is None:
            raise SkillPromotionRegistryError("invalid_hash")
        if not isinstance(maturity, str) or maturity not in _APPROVED_MATURITIES:
            raise SkillPromotionRegistryError("invalid_maturity")
        identity = (source_path, content_hash)
        if identity in loaded:
            raise SkillPromotionRegistryError("duplicate_identity")
        loaded[identity] = maturity

    return MappingProxyType(loaded)


def approved_skill_maturity(
    registry: Mapping[tuple[str, str], ApprovedMaturity],
    source_path: str,
    content_hash: str,
) -> Literal["draft", "validated", "production"]:
    """Return approved maturity only for an exact safe path-and-hash match."""
    if not _is_safe_source_path(source_path) or _HASH_RE.fullmatch(content_hash) is None:
        return "draft"
    return registry.get((source_path, content_hash), "draft")


def _load_tracked_registry() -> Mapping[tuple[str, str], ApprovedMaturity]:
    try:
        raw = Path(__file__).with_name("skill_promotions.json").read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillPromotionRegistryError("io_error") from exc
    return load_skill_promotions(raw)


SKILL_PROMOTIONS = _load_tracked_registry()
