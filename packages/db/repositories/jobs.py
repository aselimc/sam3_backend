"""Jobs repository.

Single source of truth for the state machine in upgrade/02-data-model.md
§State machine. Every transition is `UPDATE ... RETURNING` gated by a
predicate; zero rows returned means another writer beat us — caller
treats that as a no-op (loser path), never retries.

Idempotency lookup uses the unique partial index `uq_jobs_idem` as the
durable backstop if Redis is wiped.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.types import GpuClass, JobState, TaskType
from packages.db.models import LOCAL_OWNER, Job


class TransitionConflict(RuntimeError):
    """Raised when an UPDATE ... RETURNING returns zero rows (lost race)."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobsRepo:
    def __init__(self, session: AsyncSession, *, owner_id: str = LOCAL_OWNER) -> None:
        self._s = session
        self._owner = owner_id

    # ── Reads ───────────────────────────────────────────────────────────
    async def get(self, job_id: uuid.UUID) -> Job | None:
        stmt = select(Job).where(and_(Job.id == job_id, Job.owner_id == self._owner))
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def get_by_idempotency_key(self, key: str) -> Job | None:
        stmt = select(Job).where(and_(Job.owner_id == self._owner, Job.idempotency_key == key))
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        state: JobState | None = None,
        task_type: TaskType | None = None,
        limit: int = 50,
        before: datetime | None = None,
    ) -> list[Job]:
        conds = [Job.owner_id == self._owner]
        if state is not None:
            conds.append(Job.state == state)
        if task_type is not None:
            conds.append(Job.task_type == task_type)
        if before is not None:
            conds.append(Job.created_at < before)
        stmt = (
            select(Job)
            .where(and_(*conds))
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(limit)
        )
        return list((await self._s.execute(stmt)).scalars().all())

    # ── Writes ──────────────────────────────────────────────────────────
    async def insert_queued(
        self,
        *,
        task_type: TaskType,
        request_payload: dict[str, Any],
        model_id: str | None = None,
        gpu_class: GpuClass | None = None,
        idempotency_key: str | None = None,
        callback_url: str | None = None,
        max_attempts: int = 3,
    ) -> Job:
        now = _utcnow()
        job = Job(
            owner_id=self._owner,
            task_type=task_type,
            model_id=model_id,
            gpu_class=gpu_class,
            state=JobState.QUEUED,
            request_payload=request_payload,
            idempotency_key=idempotency_key,
            callback_url=callback_url,
            max_attempts=max_attempts,
            created_at=now,
            queued_at=now,
        )
        self._s.add(job)
        try:
            await self._s.flush()
        except IntegrityError as e:
            await self._s.rollback()
            raise IntegrityError("idempotency_conflict", e.params, e.orig) from e
        return job

    async def transition(
        self,
        job_id: uuid.UUID,
        *,
        from_: JobState | tuple[JobState, ...],
        to: JobState,
        celery_task_id: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        result_summary: dict[str, Any] | None = None,
        bump_attempt: bool = False,
    ) -> Job:
        """SQL-guarded transition. Raises TransitionConflict if no row matched."""
        from_states = (from_,) if isinstance(from_, JobState) else from_
        now = _utcnow()
        values: dict[str, Any] = {"state": to}
        if to == JobState.RUNNING:
            values["started_at"] = now
        if to in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED):
            values["finished_at"] = now
        if error_code is not None:
            values["error_code"] = error_code
        if error_detail is not None:
            values["error_detail"] = error_detail
        if result_summary is not None:
            values["result_summary"] = result_summary
        if bump_attempt:
            values["attempt"] = Job.attempt + 1

        conds = [
            Job.id == job_id,
            Job.owner_id == self._owner,
            Job.state.in_([s.value for s in from_states]),
        ]
        if celery_task_id is not None:
            conds.append(Job.celery_task_id == celery_task_id)

        stmt = update(Job).where(and_(*conds)).values(**values).returning(Job)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise TransitionConflict(
                f"job {job_id} not in {[s.value for s in from_states]} (or celery_task_id mismatch)"
            )
        return row

    async def claim(self, job_id: uuid.UUID, celery_task_id: str) -> Job:
        """Claim a QUEUED row by binding it to a Celery task id, then RUNNING.

        Two-step: bind first (still QUEUED), then transition. The bind step
        is the race winner; the transition runs without contention.
        """
        bind = (
            update(Job)
            .where(
                and_(
                    Job.id == job_id,
                    Job.owner_id == self._owner,
                    Job.state == JobState.QUEUED,
                    Job.celery_task_id.is_(None),
                )
            )
            .values(celery_task_id=celery_task_id)
            .returning(Job)
        )
        row = (await self._s.execute(bind)).scalar_one_or_none()
        if row is None:
            raise TransitionConflict(f"job {job_id} not claimable")
        return await self.transition(
            job_id, from_=JobState.QUEUED, to=JobState.RUNNING, celery_task_id=celery_task_id
        )

    async def heartbeat(self, job_id: uuid.UUID) -> None:
        now = _utcnow()
        await self._s.execute(
            update(Job)
            .where(and_(Job.id == job_id, Job.owner_id == self._owner))
            .values(heartbeat_at=now)
        )

    async def bump_gpu_class(self, job_id: uuid.UUID, new_class: GpuClass) -> None:
        await self._s.execute(
            update(Job)
            .where(and_(Job.id == job_id, Job.owner_id == self._owner))
            .values(gpu_class=new_class)
        )
