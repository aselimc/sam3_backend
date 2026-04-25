"""WebhookRepo — backoff schedule + dead-letter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.types import TaskType
from packages.db.repositories.jobs import JobsRepo
from packages.db.repositories.webhook import WebhookRepo


@pytest.mark.asyncio
async def test_enqueue_due_immediately(session: AsyncSession) -> None:
    job = await JobsRepo(session).insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
    )
    await session.commit()
    repo = WebhookRepo(session)
    d = await repo.enqueue(job_id=job.id, url="https://hook")
    await session.commit()
    due = await repo.due()
    assert any(x.id == d.id for x in due)


@pytest.mark.asyncio
async def test_mark_delivered_clears_retry(session: AsyncSession) -> None:
    job = await JobsRepo(session).insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
    )
    await session.commit()
    repo = WebhookRepo(session)
    d = await repo.enqueue(job_id=job.id, url="https://hook")
    await session.commit()

    await repo.mark_delivered(d.id, status_code=200)
    await session.commit()
    due = await repo.due()
    assert all(x.id != d.id for x in due)


@pytest.mark.asyncio
async def test_schedule_retry_uses_backoff(session: AsyncSession) -> None:
    job = await JobsRepo(session).insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
    )
    await session.commit()
    repo = WebhookRepo(session)
    d = await repo.enqueue(job_id=job.id, url="https://hook")
    await session.commit()

    await repo.schedule_retry(d.id, status_code=500, response_body="oops")
    await session.commit()
    fetched = await session.get(type(d), d.id)
    assert fetched is not None
    assert fetched.attempt == 1
    assert fetched.status_code == 500
    # next_retry_at is roughly now + 1 s for attempt=1
    assert fetched.next_retry_at is not None
    delta = fetched.next_retry_at - datetime.now(UTC)
    assert delta < timedelta(seconds=10)


@pytest.mark.asyncio
async def test_dead_letter_after_max_attempts(session: AsyncSession) -> None:
    job = await JobsRepo(session).insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
    )
    await session.commit()
    repo = WebhookRepo(session)
    d = await repo.enqueue(job_id=job.id, url="https://hook")
    await session.commit()

    for _ in range(WebhookRepo.MAX_ATTEMPTS + 1):
        await repo.schedule_retry(d.id, status_code=500, response_body="x")
    await session.commit()
    fetched = await session.get(type(d), d.id)
    assert fetched is not None
    assert fetched.next_retry_at is None  # dead-lettered
    assert fetched.attempt == WebhookRepo.MAX_ATTEMPTS + 1
