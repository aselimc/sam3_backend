"""Distributed locks on Redis.

`redis_lock(key, ttl)` is a context manager that performs `SET NX PX`
with a random fencing token, and releases via Lua so a stale holder
cannot delete a fresh lock. Used for:

- `lock:model:{model_id}` — first-load coordination across workers (10 min)
- `lock:beat` — Celery-beat leader election (30 s)

See upgrade/02-data-model.md §Redis layout.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager

from redis import Redis

# Lua: delete only if the value matches our token. KEYS[1]=key, ARGV[1]=token.
_RELEASE_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""


class LockNotAcquired(RuntimeError):
    pass


@contextmanager
def redis_lock(
    redis: Redis,
    key: str,
    ttl_ms: int,
    *,
    blocking: bool = False,
    raise_on_failure: bool = True,
) -> Iterator[bool]:
    """Acquire `key` with a fencing token; release iff still ours.

    Yields True if acquired, False otherwise. With `blocking=False` (default),
    a single SET NX attempt is made — callers retry at their cadence.
    """
    token = secrets.token_hex(16)
    acquired = bool(redis.set(name=key, value=token, nx=True, px=ttl_ms))
    if not acquired and raise_on_failure and not blocking:
        raise LockNotAcquired(key)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                redis.eval(_RELEASE_LUA, 1, key, token)
            except Exception:
                # Lock will expire on its own; never raise from finally.
                pass
