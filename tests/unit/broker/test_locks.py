"""redis_lock acquires with NX/PX, releases with token-matching Lua."""

from __future__ import annotations

import pytest

from packages.broker.locks import LockNotAcquired, redis_lock


def test_acquire_release_round_trip(redis) -> None:
    with redis_lock(redis, "lock:k", ttl_ms=1000) as held:
        assert held is True
        assert redis.get("lock:k") is not None
    assert redis.get("lock:k") is None


def test_second_acquire_fails_while_held(redis) -> None:
    with redis_lock(redis, "lock:k", ttl_ms=1000):
        with pytest.raises(LockNotAcquired):
            with redis_lock(redis, "lock:k", ttl_ms=1000):
                pass


def test_blocking_false_returns_false_when_taken(redis) -> None:
    with redis_lock(redis, "lock:k", ttl_ms=1000):
        with redis_lock(
            redis, "lock:k", ttl_ms=1000, blocking=True, raise_on_failure=False
        ) as held:
            assert held is False


def test_release_only_deletes_own_token(redis) -> None:
    with redis_lock(redis, "lock:k", ttl_ms=5000):
        # Simulate a stale holder overwriting the value mid-flight; the
        # release Lua must NOT delete it because the token mismatches.
        redis.set("lock:k", "imposter")
    assert redis.get("lock:k") == b"imposter"


def test_ttl_set(redis) -> None:
    with redis_lock(redis, "lock:k", ttl_ms=2000):
        ttl = redis.pttl("lock:k")
        assert 0 < ttl <= 2000
