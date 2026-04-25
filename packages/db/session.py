"""Async SQLAlchemy session factory.

One engine per process; one session per request/task. The factory takes
a URL so tests can pass `sqlite+aiosqlite:///:memory:`. Pool sizes come
from Settings; use `NullPool` under SQLite to avoid the "different
connection" pitfall.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from packages.core.config import get_settings


def make_engine(url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    db_url = url or get_settings().database_url
    kwargs: dict = {"echo": echo, "future": True}
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        # In-memory SQLite needs StaticPool so every session sees the same DB.
        # File-backed SQLite uses NullPool to dodge cross-connection locking.
        kwargs["poolclass"] = StaticPool if ":memory:" in db_url else NullPool
    return create_async_engine(db_url, **kwargs)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        _engine = make_engine()
        _session_factory = make_session_factory(_engine)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


def async_session_maker() -> async_sessionmaker[AsyncSession]:
    """Backwards-compatible alias matching upgrade/12 wording."""
    return get_session_factory()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Open a session that commits on clean exit, rolls back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def reset_engine() -> None:
    """Test helper — drop the cached engine so the next call re-reads settings."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
