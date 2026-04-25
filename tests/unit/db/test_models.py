"""Schema smoke — Base.metadata.create_all + insert each table."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.types import JobState, TaskType
from packages.db.models import Artifact, Job, JobEvent, WebhookDelivery


@pytest.mark.asyncio
async def test_create_all_emits_all_tables(session: AsyncSession) -> None:
    job = Job(task_type=TaskType.SEGMENTATION_TEXT, request_payload={"x": 1})
    session.add(job)
    await session.flush()
    session.add(JobEvent(job_id=job.id, event_type="state", data={"state": "QUEUED"}))
    session.add(
        Artifact(
            job_id=job.id,
            storage_key="s3://b/k",
            role="depth.png",
            content_type="image/png",
            size_bytes=10,
        )
    )
    session.add(WebhookDelivery(job_id=job.id, url="https://hook"))
    await session.commit()

    rows = (await session.execute(select(Job))).scalars().all()
    assert len(rows) == 1
    assert rows[0].owner_id == "local"
    assert rows[0].state == JobState.QUEUED


@pytest.mark.asyncio
async def test_idempotency_unique_per_owner(session: AsyncSession) -> None:
    j1 = Job(task_type=TaskType.DEPTH_MONOCULAR, idempotency_key="k1")
    session.add(j1)
    await session.commit()

    j2 = Job(task_type=TaskType.DEPTH_MONOCULAR, idempotency_key="k1")
    session.add(j2)
    with pytest.raises(Exception):
        await session.commit()
    await session.rollback()


@pytest.mark.asyncio
async def test_idempotency_nullable(session: AsyncSession) -> None:
    session.add(Job(task_type=TaskType.DEPTH_MONOCULAR, id=uuid.uuid4()))
    session.add(Job(task_type=TaskType.DEPTH_MONOCULAR, id=uuid.uuid4()))
    await session.commit()
    assert (await session.execute(select(Job))).scalars().all().__len__() == 2
