"""Integration: JobsRepo state guard against real Postgres.

Two concurrent claims race for the same QUEUED row; only one survives.
This is the loser-path that the unit suite simulates sequentially.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from packages.core.types import JobState, TaskType
from packages.db.repositories.jobs import JobsRepo, TransitionConflict
from packages.db.session import make_session_factory


@pytest.mark.asyncio
async def test_concurrent_claim_only_one_winner(engine: AsyncEngine) -> None:
    factory = make_session_factory(engine)
    async with factory() as setup:
        job = await JobsRepo(setup).insert_queued(
            task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
        )
        await setup.commit()

    async def attempt(tid: str) -> bool:
        async with factory() as s:
            try:
                await JobsRepo(s).claim(job.id, celery_task_id=tid)
                await s.commit()
                return True
            except TransitionConflict:
                await s.rollback()
                return False

    results = await asyncio.gather(*(attempt(f"t-{i}") for i in range(8)))
    assert results.count(True) == 1, results
    assert results.count(False) == 7

    async with factory() as s:
        winner = await JobsRepo(s).get(job.id)
    assert winner is not None
    assert winner.state == JobState.RUNNING
    assert winner.celery_task_id is not None and winner.celery_task_id.startswith("t-")
