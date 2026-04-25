"""Presign wrapper.

Thin call to a `StorageBackend`. Centralizes the TTL caps from
upgrade/06 §Presigning (PUT 15 min, GET 10 min, hard cap 1 h) so router
code does not duplicate the policy. `now()` is injected for tests with a
fake clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from packages.storage.base import (
    PresignedDownload,
    PresignedUpload,
    StorageBackend,
)

DEFAULT_PUT_TTL_S = 15 * 60
DEFAULT_GET_TTL_S = 10 * 60
HARD_TTL_CAP_S = 60 * 60


def _capped(ttl_s: int) -> int:
    return max(1, min(ttl_s, HARD_TTL_CAP_S))


class Presigner:
    def __init__(
        self,
        storage: StorageBackend,
        *,
        now_s: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._storage = storage
        self._now = now_s

    def put(
        self,
        bucket: str,
        key: str,
        *,
        content_type: str,
        content_length: int,
        ttl_s: int = DEFAULT_PUT_TTL_S,
    ) -> PresignedUpload:
        return self._storage.presign_put(
            bucket,
            key,
            content_type=content_type,
            content_length=content_length,
            ttl_s=_capped(ttl_s),
        )

    def get(
        self,
        bucket: str,
        key: str,
        *,
        ttl_s: int = DEFAULT_GET_TTL_S,
        response_content_disposition: str | None = None,
    ) -> PresignedDownload:
        return self._storage.presign_get(
            bucket,
            key,
            ttl_s=_capped(ttl_s),
            response_content_disposition=response_content_disposition,
        )

    def now_s(self) -> int:
        return self._now()
