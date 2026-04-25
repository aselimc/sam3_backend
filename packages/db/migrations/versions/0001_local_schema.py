"""local profile baseline schema

Revision ID: 0001_local_schema
Revises:
Create Date: 2026-04-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0001_local_schema"
down_revision: str | None = None
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    return op.get_context().dialect.name == "postgresql"


def upgrade() -> None:
    json_t = sa.dialects.postgresql.JSONB if _is_postgres() else sa.JSON

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("owner_id", sa.Text(), nullable=False, server_default="local"),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=True),
        sa.Column("gpu_class", sa.String(32), nullable=True),
        sa.Column("state", sa.String(16), nullable=False, server_default="QUEUED"),
        sa.Column("request_payload", json_t(), nullable=True),
        sa.Column("result_summary", json_t(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.Text(), nullable=True),
        sa.Column("callback_url", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("gpu_seconds_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("output_bytes", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.create_index("idx_jobs_owner_id", "jobs", ["owner_id"])
    op.create_index("idx_jobs_owner_created", "jobs", ["owner_id", "created_at"])

    if _is_postgres():
        op.execute(
            "CREATE INDEX idx_jobs_state ON jobs (state) "
            "WHERE state IN ('QUEUED','RUNNING','RETRYING','CANCELING')"
        )
        op.execute(
            "CREATE UNIQUE INDEX uq_jobs_idem ON jobs (owner_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )
        op.execute(
            "CREATE INDEX idx_jobs_heartbeat ON jobs (heartbeat_at) "
            "WHERE state IN ('RUNNING','CANCELING')"
        )
    else:
        op.create_index("idx_jobs_state", "jobs", ["state"])
        op.create_index(
            "uq_jobs_idem",
            "jobs",
            ["owner_id", "idempotency_key"],
            unique=True,
        )
        op.create_index("idx_jobs_heartbeat", "jobs", ["heartbeat_at"])

    op.create_table(
        "job_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("data", json_t(), nullable=True),
        sa.Column(
            "at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_job_events_job_id", "job_events", ["job_id"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("checksum_sha256", sa.String(64), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_artifacts_job", "artifacts", ["job_id"])

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_webhook_job_id", "webhook_deliveries", ["job_id"])
    if _is_postgres():
        op.execute(
            "CREATE INDEX idx_webhook_retry_due ON webhook_deliveries (next_retry_at) "
            "WHERE delivered_at IS NULL"
        )
    else:
        op.create_index("idx_webhook_retry_due", "webhook_deliveries", ["next_retry_at"])


def downgrade() -> None:
    op.drop_index("idx_webhook_retry_due", table_name="webhook_deliveries")
    op.drop_index("idx_webhook_job_id", table_name="webhook_deliveries")
    op.drop_table("webhook_deliveries")
    op.drop_index("idx_artifacts_job", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index("idx_job_events_job_id", table_name="job_events")
    op.drop_table("job_events")
    op.drop_index("idx_jobs_heartbeat", table_name="jobs")
    op.drop_index("uq_jobs_idem", table_name="jobs")
    op.drop_index("idx_jobs_state", table_name="jobs")
    op.drop_index("idx_jobs_owner_created", table_name="jobs")
    op.drop_index("idx_jobs_owner_id", table_name="jobs")
    op.drop_table("jobs")
