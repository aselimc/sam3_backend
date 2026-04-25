"""JobsRepo — state guard + idempotency + lost-race path."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from packages.core.types import GpuClass, JobState, TaskType
from packages.db.repositories.jobs import JobsRepo, TransitionConflict
from packages.db.session import make_session_factory


@pytest.mark.asyncio
async def test_insert_queued_then_get(session: AsyncSession) -> None:
    repo = JobsRepo(session)
    job = await repo.insert_queued(
        task_type=TaskType.SEGMENTATION_TEXT,
        request_payload={"queries": [{"text": "cat"}]},
    )
    await session.commit()
    fetched = await repo.get(job.id)
    assert fetched is not None
    assert fetched.state == JobState.QUEUED
    assert fetched.queued_at is not None


@pytest.mark.asyncio
async def test_idempotency_lookup(session: AsyncSession) -> None:
    repo = JobsRepo(session)
    job = await repo.insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR,
        request_payload={},
        idempotency_key="abc",
    )
    await session.commit()
    found = await repo.get_by_idempotency_key("abc")
    assert found is not None and found.id == job.id


@pytest.mark.asyncio
async def test_transition_happy_path(session: AsyncSession) -> None:
    repo = JobsRepo(session)
    job = await repo.insert_queued(task_type=TaskType.DEPTH_MONOCULAR, request_payload={})
    await session.commit()

    await repo.claim(job.id, celery_task_id="t1")
    await session.commit()

    after = await repo.get(job.id)
    assert after is not None
    assert after.state == JobState.RUNNING
    assert after.celery_task_id == "t1"
    assert after.started_at is not None

    await repo.transition(
        job.id,
        from_=JobState.RUNNING,
        to=JobState.SUCCEEDED,
        celery_task_id="t1",
        result_summary={"ok": True},
    )
    await session.commit()
    final = await repo.get(job.id)
    assert final is not None and final.state == JobState.SUCCEEDED
    assert final.finished_at is not None


@pytest.mark.asyncio
async def test_transition_loser_path(session: AsyncSession) -> None:
    repo = JobsRepo(session)
    job = await repo.insert_queued(task_type=TaskType.DEPTH_MONOCULAR, request_payload={})
    await session.commit()
    # Wrong celery_task_id — guard rejects
    with pytest.raises(TransitionConflict):
        await repo.transition(
            job.id, from_=JobState.QUEUED, to=JobState.RUNNING, celery_task_id="never-ran"
        )


@pytest.mark.asyncio
async def test_second_claim_loses_race(engine: AsyncEngine) -> None:
    """Sequential simulation of two workers claiming the same QUEUED row.

    The bind step is `UPDATE ... WHERE state='QUEUED' AND celery_task_id
    IS NULL`. After the first commit the predicate fails for any other
    task id — second caller hits TransitionConflict, never runs.
    """
    factory = make_session_factory(engine)
    async with factory() as setup:
        repo = JobsRepo(setup)
        job = await repo.insert_queued(task_type=TaskType.DEPTH_MONOCULAR, request_payload={})
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

    # Sequential — the second call MUST lose because celery_task_id is bound.
    assert await attempt("a") is True
    assert await attempt("b") is False
    _ = asyncio  # keep import; concurrent variant covered by integration test


@pytest.mark.asyncio
async def test_bump_gpu_class(session: AsyncSession) -> None:
    repo = JobsRepo(session)
    job = await repo.insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}, gpu_class=GpuClass.T4_16G
    )
    await session.commit()
    await repo.bump_gpu_class(job.id, GpuClass.A100_40G)
    await session.commit()
    fetched = await repo.get(job.id)
    assert fetched is not None and fetched.gpu_class == GpuClass.A100_40G
