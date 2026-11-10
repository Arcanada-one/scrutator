"""FI10 public/private Scrutator namespaces (DEC-AUP-0161 family).

Synthetic-first product surface for FI10-T5.2: namespace names, ACL policy,
generation readback and withdrawal purge. Live Feeder grant / production
namespace enablement remains HELD (paired with T5.4 + an accepted generation).
"""

from scrutator.fi10.namespaces import (
    FI10_NAMESPACES,
    FI10_PRIVATE,
    FI10_PUBLIC,
    AclDecision,
    decide_read,
)
from scrutator.fi10.readback import GenerationReadback, readback_generation
from scrutator.fi10.store import SyntheticFi10Store
from scrutator.fi10.withdrawal import PurgeResult, purge_withdrawn, resolve_serving_generation

__all__ = [
    "FI10_PRIVATE",
    "FI10_PUBLIC",
    "FI10_NAMESPACES",
    "AclDecision",
    "decide_read",
    "GenerationReadback",
    "readback_generation",
    "SyntheticFi10Store",
    "PurgeResult",
    "purge_withdrawn",
    "resolve_serving_generation",
]
