from concurrent.futures import ThreadPoolExecutor

from scrutator.request_rate_limit import SlidingWindowRateLimiter


def test_sliding_window_returns_retry_after_and_expires_oldest_event():
    now = [100.0]
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60, clock=lambda: now[0])

    assert limiter.try_acquire("kc2-store") is None
    now[0] = 110.0
    assert limiter.try_acquire("kc2-store") is None
    assert limiter.try_acquire("kc2-store") == 50

    now[0] = 160.0
    assert limiter.try_acquire("kc2-store") is None


def test_concurrent_attempts_never_exceed_limit():
    limiter = SlidingWindowRateLimiter(limit=7, window_seconds=60, clock=lambda: 100.0)

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda _index: limiter.try_acquire("kc2-store"), range(50)))

    assert results.count(None) == 7
    assert all(retry == 60 for retry in results if retry is not None)


def test_invalid_rate_limit_configuration_is_refused():
    try:
        SlidingWindowRateLimiter(limit=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero limit must be rejected")

    try:
        SlidingWindowRateLimiter(window_seconds=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero window must be rejected")


def test_limiter_refuses_unexpected_second_scope():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    assert limiter.try_acquire("kc2-store") is None
    try:
        limiter.try_acquire("other")
    except ValueError as exc:
        assert "one namespace" in str(exc)
    else:
        raise AssertionError("a second scope must not create an unbounded bucket")


def test_clock_is_sampled_after_waiting_for_rate_limit_lock():
    import threading

    class ObservedLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.enter_attempted = threading.Event()

        def acquire(self):
            self._lock.acquire()

        def release(self):
            self._lock.release()

        def __enter__(self):
            self.enter_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self._lock.release()

    now = [100.0]
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60, clock=lambda: now[0])
    lock = ObservedLock()
    limiter._lock = lock

    lock.acquire()
    lock_is_held = True
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pending = pool.submit(limiter.try_acquire, "kc2-store")
        assert lock.enter_attempted.wait(timeout=2)
        now[0] = 200.0
        lock.release()
        lock_is_held = False
        assert pending.result(timeout=2) is None
    finally:
        # Release before joining so a failed assertion cannot strand the worker.
        if lock_is_held:
            lock.release()
        pool.shutdown(wait=True)

    # If time was sampled before waiting for the lock, the stale event at 100
    # expires immediately and incorrectly grants this second request.
    assert limiter.try_acquire("kc2-store") == 60
