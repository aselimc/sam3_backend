"""Static API key validator (local profile).

Single shared key compared in constant time. Resolves to the constant
`Principal(owner_id="local", scopes=["*"])`. The enterprise overlay
swaps this for JWT + per-key scopes — both paths land on the same
Principal shape so routers do not change.

See upgrade/06 §Identity, upgrade/03 §Authentication.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field

from packages.core.config import Settings, get_settings
from packages.core.errors import Unauthorized

LOCAL_OWNER = "local"


@dataclass(frozen=True, slots=True)
class Principal:
    owner_id: str
    scopes: tuple[str, ...] = ("*",)
    auth_method: str = "api_key"

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


@dataclass(frozen=True, slots=True)
class _LocalSettings:
    """Snapshot of just the auth-relevant fields for clarity in tests."""

    local_api_key: str = field(default="")


def verify_local_key(presented: str | None, *, settings: Settings | None = None) -> Principal:
    """Constant-time compare; raise Unauthorized on mismatch or missing key."""
    if not presented:
        raise Unauthorized("missing X-API-Key")
    s = settings or get_settings()
    expected = s.local_api_key
    if not expected or not hmac.compare_digest(expected.encode(), presented.encode()):
        raise Unauthorized("invalid X-API-Key")
    return Principal(owner_id=LOCAL_OWNER, scopes=("*",), auth_method="api_key")
