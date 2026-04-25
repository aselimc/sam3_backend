"""Integration fixtures for the DB against the compose Postgres.

Skips the whole module when DATABASE_URL_TEST is unreachable. Uses an
ephemeral schema named `it_<8 hex>` so concurrent runs do not collide.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from packages.db.models import Base
from packages.db.session import make_engine, make_session_factory

DB_TEST_URL = os.environ.get(
    "DATABASE_URL_TEST",
    "postgresql+asyncpg://sam3:sam3@localhost:5432/sam3",
)


def _ping(url: str) -> bool:
    async def _go() -> bool:
        eng = make_engine(url)
        try:
            async with eng.connect() as c:
                await c.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False
        except Exception:
            return False
        finally:
            await eng.dispose()

    try:
        return asyncio.run(_go())
    except Exception:
        return False


_HAS_PG = _ping(DB_TEST_URL)


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_pg() -> None:
    if not _HAS_PG:
        pytest.skip(f"compose Postgres unreachable at {DB_TEST_URL}", allow_module_level=True)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    schema = f"it_{uuid.uuid4().hex[:8]}"
    eng = make_engine(DB_TEST_URL)
    async with eng.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.execute(text(f'SET search_path TO "{schema}"'))
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        async with eng.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await eng.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = make_session_factory(engine)
    async with factory() as s:
        yield s
