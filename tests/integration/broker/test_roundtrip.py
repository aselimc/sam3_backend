"""End-to-end broker round-trip vs compose Redis.

Submits a Celery task with W3C trace context, consumes it eagerly,
verifies state-of-the-world: payload arrived intact, traceparent
survived the hop, lock + rate limiter + pub/sub all behave against a
real Redis (not fakeredis). See upgrade/12 step 2.8.
"""

from __future__ import annotations

import threading
import time
import uuid

from celery import Celery
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from packages.broker.celery_app import build_celery
from packages.broker.locks import LockNotAcquired, redis_lock
from packages.broker.pubsub import publish_event, subscribe
from packages.broker.ratelimit import RateLimiter
from packages.broker.trace import extract_from_payload, submit_with_trace
from packages.core.config import Settings


def _make_app(redis_url: str, keyprefix: str) -> Celery:
    s = Settings(
        celery_broker_url=redis_url,
        celery_result_backend=redis_url,
        celery_keyprefix=keyprefix,
    )
    app = build_celery(s)
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    return app


def test_send_task_with_trace_round_trip(redis_url: str, keyprefix: str) -> None:
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())

    app = _make_app(redis_url, keyprefix)
    seen: dict[str, object] = {}

    @app.task(name="sam3.test.run_task")
    def run_task(payload: dict) -> str:
        seen["payload"] = payload
        ctx = extract_from_payload(payload)
        seen["span_valid"] = trace.get_current_span(ctx).get_span_context().is_valid
        return "ok"

    tracer = trace.get_tracer("integration")
    payload = {"job_id": str(uuid.uuid4())}
    with tracer.start_as_current_span("api.tasks.submit"):
        result = submit_with_trace(app, "sam3.test.run_task", payload, queue="task.default")

    assert result.get(timeout=2) == "ok"
    assert seen["payload"]["job_id"] == payload["job_id"]
    assert "traceparent" in seen["payload"]
    assert seen["span_valid"] is True


def test_lock_releases_after_block(redis, keyprefix: str) -> None:
    key = f"{keyprefix}lock:model:test"
    with redis_lock(redis, key, ttl_ms=2000):
        try:
            with redis_lock(redis, key, ttl_ms=500):
                raise AssertionError("should not acquire")
        except LockNotAcquired:
            pass
    # released → acquirable again
    with redis_lock(redis, key, ttl_ms=500) as held:
        assert held is True


def test_ratelimit_burst_then_refill(redis, keyprefix: str) -> None:
    rl = RateLimiter(redis, burst=4, refill_per_sec=200.0)
    bucket = f"{keyprefix}rl:owner:default"
    allowed = sum(1 for _ in range(8) if rl.acquire(bucket).allowed)
    # First 4 within burst should pass; refill at 200/s gives extras across the loop.
    assert 4 <= allowed <= 8
    time.sleep(0.05)
    assert rl.acquire(bucket).allowed is True


def test_pubsub_round_trip(redis, keyprefix: str) -> None:
    job_id = f"{keyprefix}j1"
    received: list[dict] = []
    started = threading.Event()

    def consume() -> None:
        gen = subscribe(redis, job_id=job_id, timeout=0.05)
        started.set()
        for ev in gen:
            received.append(ev)
            gen.close()
            return

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    started.wait(timeout=1.0)
    time.sleep(0.05)

    publish_event(redis, job_id, "RUNNING", attempt=1)
    t.join(timeout=2.0)
    assert received and received[0]["state"] == "RUNNING"
