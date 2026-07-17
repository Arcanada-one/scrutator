"""Route-specific machine capabilities for non-reader mutation paths."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException

from scrutator.config import settings


@dataclass(frozen=True)
class NamespaceCapability:
    namespaces: frozenset[str]
    operator: bool = False


def _namespaces(value: str) -> frozenset[str]:
    return frozenset(name.strip() for name in value.split(",") if name.strip())


def _matches(configured: str, presented: str | None) -> bool:
    try:
        configured_bytes = configured.encode("ascii")
        presented_bytes = presented.encode("ascii") if presented else b""
    except UnicodeEncodeError:
        return False
    return (
        bool(configured_bytes) and bool(presented_bytes) and secrets.compare_digest(configured_bytes, presented_bytes)
    )


async def require_feeder_capability(
    x_kb_feeder_token: str | None = Header(default=None),
) -> NamespaceCapability:
    if not _matches(settings.feeder_token, x_kb_feeder_token):
        raise HTTPException(status_code=401, detail="feeder credential required")
    return NamespaceCapability(namespaces=_namespaces(settings.feeder_namespaces))


async def require_rollback_capability(
    x_kb_rollback_token: str | None = Header(default=None),
) -> NamespaceCapability:
    scheduled = _matches(settings.rollback_token, x_kb_rollback_token)
    operator = _matches(settings.operator_rollback_token, x_kb_rollback_token)
    if not scheduled and not operator:
        raise HTTPException(status_code=401, detail="rollback credential required")
    return NamespaceCapability(
        namespaces=_namespaces(settings.rollback_namespaces),
        operator=operator,
    )


async def require_ltm_writer_capability(
    x_ltm_writer_token: str | None = Header(default=None),
) -> NamespaceCapability:
    if not _matches(settings.ltm_writer_token, x_ltm_writer_token):
        raise HTTPException(status_code=401, detail="LTM writer credential required")
    return NamespaceCapability(namespaces=_namespaces(settings.ltm_writer_namespaces))
