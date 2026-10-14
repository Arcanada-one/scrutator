"""In-process sliding-window limiter for the singleton exact-evidence writer route.

The production deploy transaction enforces one Scrutator container, and the
container starts one Uvicorn process. This limiter is therefore intentionally
process-local. If that deployment contract changes to multiple workers or
replicas, replace this with shared atomic state before relying on a global cap.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

EVIDENCE_EXACT_REQUEST_LIMIT = 120
EVIDENCE_EXACT_WINDOW_SECONDS = 60.0


class SlidingWindowRateLimiter:
    """Allow a fixed number of attempts per rolling window.

    One instance is dedicated to one route and its server-selected singleton
    namespace, so state is one bounded deque rather than caller-controlled
    buckets. The caller must authenticate and authorize scope before consuming
    a slot.
    """

    def __init__(
        self,
        *,
        limit: int = EVIDENCE_EXACT_REQUEST_LIMIT,
        window_seconds: float = EVIDENCE_EXACT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._key: str | None = None
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> int | None:
        """Record an authorized attempt or return Retry-After seconds if full."""
        if not key:
            raise ValueError("rate-limit key must be non-empty")
        now = self._clock()
        with self._lock:
            if self._key is None:
                self._key = key
            elif self._key != key:
                raise ValueError("limiter instance is scoped to one namespace")
            cutoff = now - self._window_seconds
            while self._events and self._events[0] <= cutoff:
                self._events.popleft()

            if len(self._events) >= self._limit:
                return max(1, math.ceil(self._events[0] + self._window_seconds - now))

            self._events.append(now)
            return None
