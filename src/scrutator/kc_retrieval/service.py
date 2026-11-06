"""Orchestrate registry lookup → role fallback → sha256 verify → encode response."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import HTTPException

from scrutator.kc_retrieval.models import Citation, KcRetrieveQuery, KcRetrieveResponse
from scrutator.kc_retrieval.registry import (
    expected_sha256,
    load_registry,
    object_kind_for,
    ordered_candidates,
    rows_for_object,
)


@dataclass(frozen=True)
class RetrievalPolicy:
    registry_path: Path
    max_bytes: int
    public_object_ids: frozenset[str]
    namespace: str


def _sha256_file(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    size = path.stat().st_size
    if size > max_bytes:
        raise HTTPException(status_code=413, detail="object exceeds size budget")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="object exceeds size budget")
            digest.update(block)
            chunks.append(block)
    return b"".join(chunks), digest.hexdigest()


def _encode_content(data: bytes) -> tuple[Literal["utf-8", "base64"], str]:
    try:
        return "utf-8", data.decode("utf-8")
    except UnicodeDecodeError:
        return "base64", base64.b64encode(data).decode("ascii")


def _authorize(object_id: str, *, allowed_namespaces: frozenset[str], policy: RetrievalPolicy) -> None:
    """Public allowlist for unscoped callers; private items need the KC namespace grant.

    Unknown and unauthorized private ids both answer 404 so the route is not an
    existence oracle for private library membership.
    """
    if object_id in policy.public_object_ids:
        return
    if policy.namespace in allowed_namespaces:
        return
    raise HTTPException(status_code=404, detail="not_found")


def retrieve(
    query: KcRetrieveQuery,
    *,
    allowed_namespaces: frozenset[str],
    policy: RetrievalPolicy,
) -> KcRetrieveResponse:
    _authorize(query.id, allowed_namespaces=allowed_namespaces, policy=policy)

    try:
        kind = object_kind_for(query.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid object id") from exc

    if query.view == "package" and kind != "package":
        raise HTTPException(status_code=422, detail="view=package requires a package id")
    if query.view == "original" and kind != "original":
        raise HTTPException(status_code=422, detail="view=original requires a source id")

    effective_view: Literal["original", "package", "fragment"]
    if query.view == "fragment":
        effective_view = "fragment"
    elif kind == "package":
        effective_view = "package"
    else:
        effective_view = "original"

    try:
        doc = load_registry(policy.registry_path)
    except Exception as exc:  # RegistryError and unexpected IO surface as unavailable
        raise HTTPException(status_code=503, detail="location registry unavailable") from exc

    rows = rows_for_object(doc, query.id)
    if not rows:
        raise HTTPException(status_code=404, detail="not_found")

    last_error: HTTPException | None = None
    for row in ordered_candidates(rows):
        declared = row.get("size_bytes")
        if isinstance(declared, int) and declared > policy.max_bytes:
            last_error = HTTPException(status_code=413, detail="object exceeds size budget")
            continue

        expected = expected_sha256(row)
        if expected is None:
            last_error = HTTPException(status_code=409, detail="copy digest not measured")
            continue

        path = Path(str(row["path"]))
        if not path.is_file():
            last_error = HTTPException(status_code=404, detail="not_found")
            continue

        try:
            raw, observed = _sha256_file(path, max_bytes=policy.max_bytes)
        except HTTPException as exc:
            last_error = exc
            continue
        except OSError:
            last_error = HTTPException(status_code=404, detail="not_found")
            continue

        if observed != expected:
            # Corrupt copy: refuse this candidate; try mirror/backup before failing closed.
            last_error = HTTPException(status_code=409, detail="corrupt_copy")
            continue

        start = 0
        end = len(raw)
        if effective_view == "fragment":
            if query.offset_start is None or query.offset_end is None:
                raise HTTPException(status_code=422, detail="fragment view requires offset_start and offset_end")
            if query.offset_end < query.offset_start or query.offset_end > len(raw):
                raise HTTPException(status_code=422, detail="fragment offsets out of range")
            start = query.offset_start
            end = query.offset_end
            payload = raw[start:end]
        else:
            payload = raw

        encoding, content = _encode_content(payload)
        role = row["role"]
        citation = Citation(
            object_id=query.id,
            object_kind=kind,
            role_served=role,
            byte_offset_start=start,
            byte_offset_end=end,
            sha256=observed,
        )
        return KcRetrieveResponse(
            object_id=query.id,
            object_kind=kind,
            role_served=role,
            sha256=observed,
            size_bytes=len(payload),
            content_encoding=encoding,
            content=content,
            citation=citation,
            view=effective_view,
            kc2_admission="not_measured",
        )

    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=404, detail="not_found")
