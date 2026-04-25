"""Integration fixtures for the broker against the compose Redis.

Skips the whole module when no Redis is reachable on `REDIS_URL`
(default `redis://localhost:6379/15` — DB 15 is the test scratch DB).
"""

from __future__ import annotations

import os
import uuid

import pytest
from redis import Redis
from redis.exceptions import RedisError

REDIS_TEST_URL = os.environ.get("BROKER_TEST_REDIS_URL", "redis://localhost:6379/15")


def _ping(url: str) -> bool:
    try:
        client = Redis.from_url(url, socket_connect_timeout=0.5)
        client.ping()
        client.close()
        return True
    except RedisError:
        return False
    except Exception:
        return False


_HAS_REDIS = _ping(REDIS_TEST_URL)


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_redis() -> None:
    if not _HAS_REDIS:
        pytest.skip(f"compose Redis unreachable at {REDIS_TEST_URL}", allow_module_level=True)


@pytest.fixture()
def redis_url() -> str:
    return REDIS_TEST_URL


@pytest.fixture()
def redis() -> Redis:
    client = Redis.from_url(REDIS_TEST_URL)
    yield client
    client.close()


@pytest.fixture()
def keyprefix() -> str:
    return f"test:{uuid.uuid4().hex[:8]}:"
