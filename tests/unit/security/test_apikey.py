"""Static API key validator."""

from __future__ import annotations

import pytest

from packages.core.config import Settings
from packages.core.errors import Unauthorized
from packages.security.apikey import LOCAL_OWNER, verify_local_key


@pytest.fixture
def settings() -> Settings:
    return Settings(local_api_key="abc-123")


def test_valid_key_returns_local_principal(settings: Settings) -> None:
    p = verify_local_key("abc-123", settings=settings)
    assert p.owner_id == LOCAL_OWNER
    assert p.has_scope("tasks:submit")
    assert p.auth_method == "api_key"


def test_invalid_key_raises(settings: Settings) -> None:
    with pytest.raises(Unauthorized):
        verify_local_key("wrong", settings=settings)


def test_missing_key_raises(settings: Settings) -> None:
    with pytest.raises(Unauthorized):
        verify_local_key(None, settings=settings)
    with pytest.raises(Unauthorized):
        verify_local_key("", settings=settings)


def test_empty_expected_key_rejects_anything(settings: Settings) -> None:
    s = Settings(local_api_key="")
    with pytest.raises(Unauthorized):
        verify_local_key("anything", settings=s)
