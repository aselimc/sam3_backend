"""SQLAlchemy 2.0 declarative models for the local profile.

Mirrors upgrade/02-data-model.md §ER diagram. Job state machine lives in
the JobState enum from `packages.core.types`. UUID PKs use the portable
`Uuid` column type so the same models compile under SQLite (tests) and
Postgres (compose + prod). JSON columns use the portable `JSON` type.

`owner_id` is `text` + default `"local"`. The enterprise overlay adds an
FK to a real users table with an additive Alembic migration; the column
stays the same shape so repository SQL does not change.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, Uuid

from packages.core.types import GpuClass, JobState, TaskType

LOCAL_OWNER = "local"


# JSONB on Postgres, JSON elsewhere — repos see the same Python dict either way.
JsonCol = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JsonCol}


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[str] = mapped_column(Text, nullable=False, default=LOCAL_OWNER, index=True)
    task_type: Mapped[TaskType] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    gpu_class: Mapped[GpuClass | None] = mapped_column(String(32), nullable=True)
    state: Mapped[JobState] = mapped_column(String(16), nullable=False, default=JobState.QUEUED)

    request_payload: Mapped[dict[str, Any] | None] = mapped_column(JsonCol, nullable=True)
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JsonCol, nullable=True)

    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    callback_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = _ts()
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    gpu_seconds_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    output_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    events: Mapped[list[JobEvent]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="raise"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="raise"
    )
    webhooks: Mapped[list[WebhookDelivery]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="raise"
    )

    __table_args__ = (
        Index("idx_jobs_owner_created", "owner_id", "created_at"),
        Index("idx_jobs_state", "state"),
        UniqueConstraint("owner_id", "idempotency_key", name="uq_jobs_idem"),
        Index("idx_jobs_heartbeat", "heartbeat_at"),
    )


class JobEvent(Base):
    __tablename__ = "job_events"

    # Integer on SQLite (so autoincrement aliases ROWID); BigInteger on Postgres.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    data: Mapped[dict[str, Any] | None] = mapped_column(JsonCol, nullable=True)
    at: Mapped[datetime] = _ts()

    job: Mapped[Job] = relationship(back_populates="events")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _ts()

    job: Mapped[Job] = relationship(back_populates="artifacts")


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped[Job] = relationship(back_populates="webhooks")

    __table_args__ = (Index("idx_webhook_retry_due", "next_retry_at"),)
