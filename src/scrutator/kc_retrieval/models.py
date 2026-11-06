"""Request/response contracts for Knowledge Contract verified-byte retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SOURCE_ID_PATTERN = r"^d_[0-9a-f]{64}$"
PACKAGE_ID_PATTERN = r"^d_[0-9a-f]{64}/[A-Za-z0-9][A-Za-z0-9._@+-]{0,119}$"
OBJECT_ID_PATTERN = rf"(?:{SOURCE_ID_PATTERN[1:-1]}|{PACKAGE_ID_PATTERN[1:-1]})"


class Citation(BaseModel):
    """Citation bound to the verified object id and optional byte range.

    Paths are intentionally omitted (LOC-R012): cards and API clients carry ids;
    the location registry alone holds host/volume/path.
    """

    object_id: str
    object_kind: Literal["original", "package"]
    role_served: Literal["primary", "mirror", "backup"]
    byte_offset_start: int = Field(ge=0)
    byte_offset_end: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KcRetrieveResponse(BaseModel):
    """Verified bytes (or a bounded fragment) for a source/package id."""

    object_id: str
    object_kind: Literal["original", "package"]
    role_served: Literal["primary", "mirror", "backup"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    content_encoding: Literal["utf-8", "base64"]
    content: str
    citation: Citation
    view: Literal["original", "package", "fragment"]
    kc2_admission: Literal["not_measured"] = "not_measured"


class KcRetrieveQuery(BaseModel):
    """Validated query parameters for GET /v1/kc/retrieve."""

    id: str
    view: Literal["original", "package", "fragment"] = "original"
    offset_start: int | None = Field(default=None, ge=0)
    offset_end: int | None = Field(default=None, ge=0)

    @field_validator("id")
    @classmethod
    def validate_object_id(cls, value: str) -> str:
        import re

        if not re.fullmatch(OBJECT_ID_PATTERN, value):
            raise ValueError("id must be a source id (d_<sha256>) or package id (d_<sha256>/<version>)")
        return value
