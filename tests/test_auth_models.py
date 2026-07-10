"""Tests for SRCH-0023 Step 1: TenantContext (src/scrutator/auth/models.py)."""

from __future__ import annotations

import pytest

from scrutator.auth.models import TenantContext


class TestTenantContext:
    def test_construction(self):
        ctx = TenantContext(
            principal_id="svc-1",
            principal_type="service",
            allowed_namespace_ids=frozenset({1, 2}),
            allowed_namespace_names=frozenset({"arcanada", "ltm-bench"}),
        )
        assert ctx.principal_id == "svc-1"
        assert ctx.principal_type == "service"
        assert ctx.allowed_namespace_ids == frozenset({1, 2})
        assert ctx.allowed_namespace_names == frozenset({"arcanada", "ltm-bench"})

    def test_frozen_immutable(self):
        """TenantContext must be frozen — mutating it after construction raises."""
        ctx = TenantContext(
            principal_id="svc-1",
            principal_type="service",
            allowed_namespace_ids=frozenset({1}),
            allowed_namespace_names=frozenset({"arcanada"}),
        )
        with pytest.raises((AttributeError, TypeError)):
            ctx.principal_id = "svc-2"

    def test_principal_type_accepts_user(self):
        ctx = TenantContext(
            principal_id="user-1",
            principal_type="user",
            allowed_namespace_ids=frozenset(),
            allowed_namespace_names=frozenset(),
        )
        assert ctx.principal_type == "user"

    def test_empty_allowed_set_is_valid(self):
        """A principal with zero grants is a valid (deny-everything) context."""
        ctx = TenantContext(
            principal_id="svc-orphan",
            principal_type="service",
            allowed_namespace_ids=frozenset(),
            allowed_namespace_names=frozenset(),
        )
        assert ctx.allowed_namespace_ids == frozenset()
