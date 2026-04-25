"""Shared fixtures for broker tests.

We use fakeredis with the Lua executor enabled so EVAL/EVALSHA round-trip
the same script the production Redis would run. Each test gets a clean
instance; no test should rely on cross-test state.
"""

from __future__ import annotations

import fakeredis
import pytest


@pytest.fixture()
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=False)
