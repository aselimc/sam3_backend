"""ArtifactsRepo basic CRUD."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.types import TaskType
from packages.db.repositories.artifacts import ArtifactsRepo
from packages.db.repositories.jobs import JobsRepo


@pytest.mark.asyncio
async def test_insert_then_list(session: AsyncSession) -> None:
    job = await JobsRepo(session).insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
    )
    await session.commit()
    repo = ArtifactsRepo(session)
    a = await repo.insert(
        job_id=job.id,
        storage_key="s3://b/depth.png",
        role="depth.png",
        content_type="image/png",
        size_bytes=42,
    )
    await session.commit()
    rows = await repo.list_for_job(job.id)
    assert len(rows) == 1
    assert rows[0].id == a.id
    assert rows[0].role == "depth.png"


@pytest.mark.asyncio
async def test_get_owner_filter(session: AsyncSession) -> None:
    job = await JobsRepo(session).insert_queued(
        task_type=TaskType.DEPTH_MONOCULAR, request_payload={}
    )
    await session.commit()
    repo_local = ArtifactsRepo(session)
    a = await repo_local.insert(job_id=job.id, storage_key="s3://b/k", role="k")
    await session.commit()

    other = ArtifactsRepo(session, owner_id="someone-else")
    assert await other.get(a.id) is None
    assert (await repo_local.get(a.id)) is not None
