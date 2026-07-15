"""Authenticated and fail-closed client for structured LTM ingest."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .secretscan import ScanError, ScanResult, scan_serialized


class ProtocolError(RuntimeError):
    """A 2xx LTM response did not satisfy the success envelope contract."""


_INGEST_COUNTS = ("entities_upserted", "edges_upserted")
_TOMBSTONE_COUNTS = (
    "chunks_deleted",
    "entity_sources_deleted",
    "edge_sources_deleted",
    "edges_deleted",
    "entities_deleted",
)


def _decode_success(response: Any, count_fields: tuple[str, ...]) -> dict[str, int | bool]:
    try:
        body = response.json()
    except Exception:
        raise ProtocolError("invalid LTM success response") from None
    if not isinstance(body, dict):
        raise ProtocolError("invalid LTM success response")
    decoded: dict[str, int | bool] = {}
    for field in count_fields:
        value = body.get(field)
        if type(value) is not int or value < 0:
            raise ProtocolError("invalid LTM success response")
        decoded[field] = value
    idempotent_noop = body.get("idempotent_noop")
    if type(idempotent_noop) is not bool:
        raise ProtocolError("invalid LTM success response")
    decoded["idempotent_noop"] = idempotent_noop
    return decoded


class LtmClient:
    def __init__(
        self,
        endpoint: str,
        writer_credential: Path,
        *,
        http: Any | None = None,
        scanner: Callable[[str], ScanResult] = scan_serialized,
    ) -> None:
        self.endpoint = endpoint
        self.writer_credential = writer_credential
        self._http = http or httpx.AsyncClient(timeout=60.0)
        self._owns_http = http is None
        self._scanner = scanner

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def _token(self) -> str:
        token = self.writer_credential.read_text().strip()
        if not token:
            raise ValueError("writer credential is empty")
        return token

    def _serialize_and_scan(self, payload: dict[str, Any]) -> bytes:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        try:
            result = self._scanner(serialized.decode())
        except Exception as exc:
            raise ScanError("secret scanner failed closed") from exc
        if result.is_critical:
            raise ScanError("secret scanner blocked outbound payload")
        return serialized

    async def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        token = self._token()
        serialized = self._serialize_and_scan(payload)
        response = await self._http.post(
            self.endpoint,
            content=serialized,
            headers={"Content-Type": "application/json", "X-LTM-Writer-Token": token},
        )
        response.raise_for_status()
        return _decode_success(response, _INGEST_COUNTS)

    async def tombstone(self, namespace: str, source_path: str) -> dict[str, int | bool]:
        token = self._token()
        serialized = self._serialize_and_scan({"namespace": namespace, "source_path": source_path})
        endpoint = self.endpoint.removesuffix("/ingest") + "/source"
        response = await self._http.request(
            "DELETE",
            endpoint,
            content=serialized,
            headers={"Content-Type": "application/json", "X-LTM-Writer-Token": token},
        )
        response.raise_for_status()
        return _decode_success(response, _TOMBSTONE_COUNTS)
