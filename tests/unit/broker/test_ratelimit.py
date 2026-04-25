"""Lua token-bucket: refill, denial after burst, concurrent-acquire count."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from packages.broker.ratelimit import RateLimiter


def test_first_acquire_allowed(redis) -> None:
    rl = RateLimiter(redis, burst=5, refill_per_sec=1.0)
    res = rl.acquire("rl:t:default")
    assert res.allowed is True
    assert res.remaining == 4
    assert res.burst == 5


def test_burst_then_deny(redis) -> None:
    rl = RateLimiter(redis, burst=3, refill_per_sec=0.0001)
    for _ in range(3):
        assert rl.acquire("rl:t:default").allowed is True
    blocked = rl.acquire("rl:t:default")
    assert blocked.allowed is False
    assert blocked.remaining == 0


def test_refill_after_sleep(redis) -> None:
    rl = RateLimiter(redis, burst=2, refill_per_sec=50.0)
    assert rl.acquire("rl:t:default").allowed is True
    assert rl.acquire("rl:t:default").allowed is True
    assert rl.acquire("rl:t:default").allowed is False
    time.sleep(0.05)  # ~2.5 tokens refilled
    assert rl.acquire("rl:t:default").allowed is True


def test_concurrent_acquire_count_matches_burst(redis) -> None:
    """N concurrent acquires return exactly `burst` allowed when refill is negligible."""
    burst = 8
    n_callers = 32
    rl = RateLimiter(redis, burst=burst, refill_per_sec=0.0001)

    def go() -> bool:
        return rl.acquire("rl:t:concurrent").allowed

    with ThreadPoolExecutor(max_workers=n_callers) as ex:
        results = list(ex.map(lambda _: go(), range(n_callers)))

    assert sum(1 for r in results if r) == burst


def test_per_call_overrides(redis) -> None:
    rl = RateLimiter(redis, burst=1, refill_per_sec=0.0001)
    res = rl.acquire("rl:t:override", burst=10, refill_per_sec=0.0001)
    assert res.allowed is True
    assert res.burst == 10
    assert res.remaining == 9


def test_evalsha_path_caches_sha(redis) -> None:
    rl = RateLimiter(redis, burst=2, refill_per_sec=0.0001)
    rl.acquire("rl:t:sha")
    sha = rl._sha
    assert sha is not None
    rl.acquire("rl:t:sha")  # would NoScriptError if SCRIPT LOAD did not stick
