"""SRCH-0038 S3 — opaque-id-only; path-like / malformed id rejected at model validation
(422) BEFORE any DB access. Path-traversal defense.

These tests construct FetchRequest directly (pure Pydantic) — no DB import is touched, which
is itself the proof that malformed input is rejected pre-DB (V-AC-6 / S3).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scrutator.db.models import FetchRequest


class TestPathLikeIdRejected:
    """V-AC-6 / S3: no selector accepts or dereferences a filesystem path."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../../etc/passwd",
            "/etc/passwd",
            "..%2f..%2fetc%2fpasswd",
            "docs/architecture.md",
            "0123456789abcdef/..",
            "0123456789ABCDEF",  # uppercase — doc ids are lowercase hex
            "0123456789abcde",  # 15 chars
            "0123456789abcdef0",  # 17 chars
            "0123456789abcdeg",  # non-hex char
            "0123456789abcdef\n../",  # trailing-newline smuggle
        ],
    )
    def test_path_like_id_rejected(self, bad_id: str):
        with pytest.raises(ValidationError):
            FetchRequest(by="source_id", id=bad_id, range="full")

    def test_document_id_selector_same_regex(self):
        with pytest.raises(ValidationError):
            FetchRequest(by="document_id", id="../../secret")


class TestMalformedIdBeforeDb:
    """A malformed id must 422 at request validation — no DB module is imported to reject it."""

    def test_malformed_id_422_before_db(self):
        # Rejection happens purely at Pydantic validation — no DB access is involved.
        with pytest.raises(ValidationError):
            FetchRequest(by="source_id", id="not-a-valid-id")

    def test_chunk_id_must_be_uuid(self):
        with pytest.raises(ValidationError):
            FetchRequest(by="chunk_id", id="0123456789abcdef")  # 16-hex is NOT a UUID
        with pytest.raises(ValidationError):
            FetchRequest(by="chunk_id", id="../../etc/passwd")

    def test_valid_doc_id_accepted(self):
        req = FetchRequest(by="source_id", id="0123456789abcdef", range="full")
        assert req.id == "0123456789abcdef"
        assert req.by == "source_id"
        assert req.range == "full"

    def test_valid_uuid_chunk_id_accepted(self):
        req = FetchRequest(by="chunk_id", id="11111111-2222-3333-4444-555555555555")
        assert req.by == "chunk_id"

    def test_parent_of_chunk_must_be_uuid(self):
        with pytest.raises(ValidationError):
            FetchRequest(by="source_id", id="0123456789abcdef", range={"parent_of_chunk": "not-a-uuid"})
        ok = FetchRequest(
            by="source_id",
            id="0123456789abcdef",
            range={"parent_of_chunk": "11111111-2222-3333-4444-555555555555"},
        )
        assert ok.range.parent_of_chunk == "11111111-2222-3333-4444-555555555555"

    def test_offset_range_bounds(self):
        with pytest.raises(ValidationError):
            FetchRequest(by="source_id", id="0123456789abcdef", range={"offset_start": 10, "offset_end": 5})
        with pytest.raises(ValidationError):
            FetchRequest(by="source_id", id="0123456789abcdef", range={"offset_start": -1, "offset_end": 5})
        ok = FetchRequest(by="source_id", id="0123456789abcdef", range={"offset_start": 3, "offset_end": 9})
        assert ok.range.offset_start == 3
        assert ok.range.offset_end == 9

    def test_extra_fields_forbidden(self):
        # S4-adjacent: closed request model — unknown keys rejected.
        with pytest.raises(ValidationError):
            FetchRequest(by="source_id", id="0123456789abcdef", injected="x")
