"""Celery factory wires the conf from upgrade/05 §Celery configuration."""

from __future__ import annotations

from packages.broker.celery_app import build_celery, get_celery
from packages.core.config import Settings


def test_factory_applies_upgrade_05_conf() -> None:
    s = Settings(
        celery_broker_url="redis://localhost:6379/1",
        celery_result_backend="redis://localhost:6379/2",
        celery_visibility_timeout=42,
        celery_keyprefix="test:",
    )
    app = build_celery(s)

    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.task_track_started is True
    assert app.conf.task_send_sent_event is True
    assert app.conf.worker_send_task_events is True
    assert app.conf.worker_cancel_long_running_tasks_on_connection_loss is True
    assert app.conf.task_max_retries == 3
    assert app.conf.task_default_retry_delay == 2

    assert app.conf.broker_url == s.celery_broker_url
    assert app.conf.result_backend == s.celery_result_backend
    assert app.conf.broker_transport_options["visibility_timeout"] == 42
    assert app.conf.broker_transport_options["global_keyprefix"] == "test:"


def test_get_celery_is_singleton() -> None:
    assert get_celery() is get_celery()
