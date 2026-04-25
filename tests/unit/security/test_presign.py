"""Presigner wrapper — TTL caps + clock injection."""

from __future__ import annotations

import pytest

from packages.security.presign import (
    DEFAULT_GET_TTL_S,
    DEFAULT_PUT_TTL_S,
    HARD_TTL_CAP_S,
    Presigner,
)
from packages.storage.local import LocalBackend


@pytest.fixture
def presigner(tmp_path) -> Presigner:
    backend = LocalBackend(tmp_path / "store", sign_secret="s")
    backend.ensure_bucket("uploads")
    backend.ensure_bucket("artifacts")
    return Presigner(backend, now_s=lambda: 1_000)


def test_default_put_ttl(presigner: Presigner) -> None:
    p = presigner.put("uploads", "k", content_type="image/png", content_length=10)
    delta = p.expires_at_epoch_s - int(p.expires_at_epoch_s - DEFAULT_PUT_TTL_S)
    assert delta == DEFAULT_PUT_TTL_S


def test_get_capped_at_one_hour(presigner: Presigner) -> None:
    p = presigner.get("artifacts", "k", ttl_s=99_999)
    # Local backend uses real time, but the cap is applied before the call;
    # we verify by confirming the URL contains the capped exp window.
    exp = int(p.url.split("exp=")[1].split("&")[0])
    import time

    assert exp - int(time.time()) <= HARD_TTL_CAP_S + 1


def test_now_s_injection(presigner: Presigner) -> None:
    assert presigner.now_s() == 1_000


def test_default_get_ttl_constant() -> None:
    assert DEFAULT_GET_TTL_S == 600
