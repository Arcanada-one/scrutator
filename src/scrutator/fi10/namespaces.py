"""Namespace constants and ACL decisions for fi10-public / fi10-private.

Read ACL is principal × namespace. Disclosure class on a record is an independent
negative control: a private-classified record must never surface from fi10-public
even if a principal somehow holds a broader grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FI10_PUBLIC = "fi10-public"
FI10_PRIVATE = "fi10-private"
FI10_NAMESPACES: frozenset[str] = frozenset({FI10_PUBLIC, FI10_PRIVATE})

DisclosureClass = Literal["public", "private"]
AclVerdict = Literal["allow", "deny"]


@dataclass(frozen=True)
class AclDecision:
    verdict: AclVerdict
    reason: str
    namespace: str
    principal_id: str


def is_fi10_namespace(name: str) -> bool:
    return name in FI10_NAMESPACES


def decide_read(
    *,
    principal_id: str,
    granted_namespaces: frozenset[str] | set[str],
    namespace: str,
    disclosure_class: DisclosureClass,
) -> AclDecision:
    """Decide whether a principal may observe a record in a requested namespace.

    Rules (synthetic + production intent):
    - Unknown FI10 namespace → deny.
    - Principal must hold an exact grant for the requested namespace (no wildcards).
    - disclosure_class=private is refused in fi10-public (positive/negative control seam).
    - disclosure_class=public is refused in fi10-private (private lane is execution-only).
    """
    if not principal_id.strip():
        return AclDecision("deny", "empty_principal", namespace, principal_id)
    if namespace not in FI10_NAMESPACES:
        return AclDecision("deny", "not_fi10_namespace", namespace, principal_id)
    if namespace not in granted_namespaces:
        return AclDecision("deny", "namespace_not_granted", namespace, principal_id)
    if namespace == FI10_PUBLIC and disclosure_class == "private":
        return AclDecision("deny", "private_disclosure_blocked_in_public", namespace, principal_id)
    if namespace == FI10_PRIVATE and disclosure_class == "public":
        return AclDecision("deny", "public_disclosure_blocked_in_private", namespace, principal_id)
    return AclDecision("allow", "granted", namespace, principal_id)
