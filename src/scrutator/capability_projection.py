"""One-way PostgreSQL authority projection into Scrutator search evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CAPABILITY_PROJECTION_MAX_REQUEST_BYTES = 8_192
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NAMESPACE_BASE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_MAX_IDENTIFIER_BYTES = 256


def _validate_identifier(value: str, label: str) -> str:
    if not value or value.strip() != value or "\0" in value:
        raise ValueError(f"{label} must be an exact non-empty string")
    if len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
        raise ValueError(f"{label} exceeds {_MAX_IDENTIFIER_BYTES} UTF-8 bytes")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a safe registry identifier")
    return value


class ProjectedCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    skill: str
    action: str
    version: int = Field(strict=True, ge=1, le=9_007_199_254_740_991)

    @field_validator("skill", "action")
    @classmethod
    def identifiers_are_exact(cls, value: str, info) -> str:
        return _validate_identifier(value, info.field_name)


class CapabilityProjectionRequest(BaseModel):
    """Closed PostgreSQL-origin projection; never an authorization input."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    source_authority: Literal["postgres"]
    tenant_id: str
    revision: int = Field(strict=True, ge=1, le=9_007_199_254_740_991)
    digest: str
    role: str
    task: str
    capability: ProjectedCapability

    @field_validator("tenant_id", "role", "task")
    @classmethod
    def identifiers_are_exact(cls, value: str, info) -> str:
        return _validate_identifier(value, info.field_name)

    @field_validator("digest")
    @classmethod
    def digest_is_lowercase_sha256(cls, value: str) -> str:
        if not _DIGEST_PATTERN.fullmatch(value):
            raise ValueError("digest must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def digest_reconciles_with_projection(self) -> CapabilityProjectionRequest:
        expected = capability_authority_digest(
            self.tenant_id,
            self.revision,
            self.role,
            self.task,
            self.capability.skill,
            self.capability.action,
            self.capability.version,
        )
        if not hmac.compare_digest(self.digest, expected):
            raise ValueError("digest does not reconcile with the PostgreSQL authority projection")
        return self


class CapabilityProjectionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    source_authority: Literal["postgres"] = "postgres"
    projection_only: Literal[True] = True
    authorization_effect: Literal["none"] = "none"
    authorization_eligible: Literal[False] = False
    tenant_id: str
    revision: int = Field(strict=True, ge=1, le=9_007_199_254_740_991)
    digest: str
    source_path: str
    namespace: str
    chunks_indexed: int = Field(strict=True, ge=1)
    strategy_used: str


def capability_authority_digest(
    tenant_id: str,
    revision: int,
    role: str,
    task: str,
    skill: str,
    action: str,
    version: int,
) -> str:
    preimage = "CapabilityAuthorityV0\0" + json.dumps(
        [tenant_id, revision, role, task, skill, action, version],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def capability_projection_tenant_key(tenant_id: str) -> str:
    return hashlib.sha256(f"CapabilityProjectionTenantV0\0{tenant_id}".encode()).hexdigest()


def capability_projection_source_path(namespace_base: str, tenant_id: str, revision: int) -> str:
    tenant_key = capability_projection_tenant_key(tenant_id)
    return f"{namespace_base}/_raw_/{tenant_key}/revision-{revision}.json"


def capability_projection_namespace(namespace_base: str, tenant_id: str) -> str:
    return f"{namespace_base}-{capability_projection_tenant_key(tenant_id)}"


def validate_capability_projection_namespace_base(value: str) -> str:
    if not _NAMESPACE_BASE_PATTERN.fullmatch(value):
        raise ValueError("capability projection namespace base is invalid")
    return value


def canonical_projection_content(request: CapabilityProjectionRequest) -> str:
    content = {
        "schema_version": 1,
        "source_authority": "postgres",
        "projection_only": True,
        "authorization_effect": "none",
        "authorization_eligible": False,
        "tenant_key": capability_projection_tenant_key(request.tenant_id),
        "revision": request.revision,
        "digest": request.digest,
        "role": request.role,
        "task": request.task,
        "capability": request.capability.model_dump(),
    }
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
