"""Celery factory wired to Redis (broker + result backend).

Conf comes from upgrade/05-worker-runtime.md §Celery configuration. Acks-late
+ reject-on-worker-lost = redelivery on crash; SQL state guard ensures
exactly-one finalizer. The factory is shared by API (send_task) and worker
(register tasks) — divergence is a bug.
"""

from __future__ import annotations

from functools import lru_cache

from celery import Celery

from packages.core.config import Settings, get_settings


def build_celery(settings: Settings | None = None) -> Celery:
    s = settings or get_settings()
    app = Celery(
        "sam3",
        broker=s.celery_broker_url,
        backend=s.celery_result_backend,
    )
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        task_send_sent_event=True,
        worker_send_task_events=True,
        worker_cancel_long_running_tasks_on_connection_loss=True,
        broker_transport_options={
            "visibility_timeout": s.celery_visibility_timeout,
            "global_keyprefix": s.celery_keyprefix,
        },
        result_expires=s.celery_result_expires,
        task_default_queue=s.celery_default_queue,
        task_default_retry_delay=2,
        task_max_retries=3,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return app


@lru_cache(maxsize=1)
def get_celery() -> Celery:
    return build_celery()


celery_app = get_celery()
