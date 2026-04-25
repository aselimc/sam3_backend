"""Artifacts repository.

Append-only: workers never overwrite an artifact in place; retries land
in `attempt=N/` siblings (see upgrade/06 §Versioning). The repo enforces
ownership via the parent Job (FK) and never returns rows whose Job row
the caller does not own.
"""

from __future__ import annotations

import uuid

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.db.models import LOCAL_OWNER, Artifact, Job


class ArtifactsRepo:
    def __init__(self, session: AsyncSession, *, owner_id: str = LOCAL_OWNER) -> None:
        self._s = session
        self._owner = owner_id

    async def insert(
        self,
        *,
        job_id: uuid.UUID,
        storage_key: str,
        role: str,
        content_type: str | None = None,
        size_bytes: int = 0,
        checksum_sha256: str | None = None,
    ) -> Artifact:
        a = Artifact(
            job_id=job_id,
            storage_key=storage_key,
            role=role,
            content_type=content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
        )
        self._s.add(a)
        await self._s.flush()
        return a

    async def list_for_job(self, job_id: uuid.UUID) -> list[Artifact]:
        stmt = (
            select(Artifact)
            .join(Job, Job.id == Artifact.job_id)
            .where(and_(Job.id == job_id, Job.owner_id == self._owner))
            .order_by(Artifact.created_at.asc(), Artifact.id.asc())
        )
        return list((await self._s.execute(stmt)).scalars().all())

    async def get(self, artifact_id: uuid.UUID) -> Artifact | None:
        stmt = (
            select(Artifact)
            .join(Job, Job.id == Artifact.job_id)
            .where(and_(Artifact.id == artifact_id, Job.owner_id == self._owner))
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()
