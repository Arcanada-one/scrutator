"""Knowledge Contract retrieval: resolve source/package id to verified bytes.

Rematch KC-PIPE-S1 store ids and KC-PIPE-S2 LocationRegistry/v1. Does not
duplicate store import or registry tooling. Live library population remains
outside this module's contract tests.
"""

from scrutator.kc_retrieval.router import router

__all__ = ["router"]
