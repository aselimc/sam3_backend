"""Integration: alembic upgrade head against compose Postgres."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import text

from packages.db.session import make_engine

DB_TEST_URL = os.environ.get(
    "DATABASE_URL_TEST",
    "postgresql+asyncpg://sam3:sam3@localhost:5432/sam3",
)
DB_TEST_URL_SYNC = DB_TEST_URL.replace("postgresql+asyncpg", "postgresql")


@pytest.mark.asyncio
async def test_alembic_upgrade_head_creates_jobs_table() -> None:
    schema = f"mig_{uuid.uuid4().hex[:8]}"
    eng = make_engine(DB_TEST_URL)
    async with eng.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    env = os.environ.copy()
    env["PGOPTIONS"] = f"-c search_path={schema}"
    # alembic uses sync driver via env.py if needed; we feed it the asyncpg URL
    # (env.py builds an async engine). Pass URL via -x arg.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            "alembic.ini",
            "-x",
            f"search_path={schema}",
            "upgrade",
            "head",
        ],
        env={
            **env,
            "DATABASE_URL": DB_TEST_URL,
        },
        capture_output=True,
        text=True,
    )
    try:
        assert proc.returncode == 0, proc.stderr
        async with eng.connect() as conn:
            await conn.execute(text(f'SET search_path TO "{schema}"'))
            res = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema=:s ORDER BY table_name"
                ),
                {"s": schema},
            )
            tables = [r[0] for r in res]
        assert {"jobs", "job_events", "artifacts", "webhook_deliveries"}.issubset(set(tables))
    finally:
        async with eng.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await eng.dispose()
