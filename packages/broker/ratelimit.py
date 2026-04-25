"""Python wrapper for the Lua token-bucket script.

Loads the script via SCRIPT LOAD on first acquire (EVALSHA + NOSCRIPT
fallback to EVAL); concurrent acquires across N tasks return a correct
allowed-count because the bucket math runs server-side and atomically.

Bucket key shape: `rl:{owner}:{bucket}` (see upgrade/02-data-model.md).
Defaults come from Settings.rate_limit_*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from redis import Redis
from redis.exceptions import NoScriptError

from packages.core.config import get_settings

_SCRIPT_PATH = Path(__file__).with_name("ratelimit.lua")


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    burst: int


class RateLimiter:
    """Token-bucket rate limiter backed by `ratelimit.lua` on Redis."""

    def __init__(
        self,
        redis: Redis,
        *,
        burst: int | None = None,
        refill_per_sec: float | None = None,
    ) -> None:
        self._redis = redis
        s = get_settings()
        self._burst = burst if burst is not None else s.rate_limit_burst
        self._refill = refill_per_sec if refill_per_sec is not None else s.rate_limit_refill_per_sec
        self._script_src = _SCRIPT_PATH.read_text(encoding="utf-8")
        self._sha: str | None = None

    def _ensure_loaded(self) -> str:
        if self._sha is None:
            self._sha = self._redis.script_load(self._script_src)
        return self._sha

    def acquire(
        self,
        bucket_key: str,
        *,
        cost: int = 1,
        burst: int | None = None,
        refill_per_sec: float | None = None,
    ) -> RateLimitResult:
        b = burst if burst is not None else self._burst
        r = refill_per_sec if refill_per_sec is not None else self._refill
        now_ms = int(time.time() * 1000)
        args = [b, r, now_ms, cost]
        try:
            sha = self._ensure_loaded()
            raw = self._redis.evalsha(sha, 1, bucket_key, *args)
        except NoScriptError:
            raw = self._redis.eval(self._script_src, 1, bucket_key, *args)
            self._sha = None
        allowed, remaining, burst_out = (int(x) for x in raw)
        return RateLimitResult(allowed=bool(allowed), remaining=remaining, burst=burst_out)
