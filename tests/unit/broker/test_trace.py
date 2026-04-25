"""W3C traceparent injects/extracts cleanly across the Celery payload."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from packages.broker.trace import extract_from_payload, otel_extract, submit_with_trace


def _ensure_provider() -> None:
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())


def test_submit_injects_traceparent_into_payload() -> None:
    _ensure_provider()
    tracer = trace.get_tracer("test")
    sent: dict[str, Any] = {}

    fake_app = MagicMock()

    def capture(name: str, *, args: list[Any], queue: str | None, **_: Any) -> str:
        sent["name"] = name
        sent["payload"] = args[0]
        sent["queue"] = queue
        return "task-id-1"

    fake_app.send_task.side_effect = capture

    payload: dict[str, Any] = {"job_id": "j1"}
    with tracer.start_as_current_span("api.tasks.submit"):
        submit_with_trace(fake_app, "run_task", payload, queue="task.default")

    assert sent["name"] == "run_task"
    assert sent["queue"] == "task.default"
    assert "traceparent" in sent["payload"]
    # span context recoverable on the worker side
    ctx = extract_from_payload(sent["payload"])
    span = trace.get_current_span(ctx)
    assert span.get_span_context().is_valid


def test_extract_with_no_trace_returns_default_context() -> None:
    ctx = extract_from_payload({"job_id": "j1"})
    span = trace.get_current_span(ctx)
    assert span.get_span_context().is_valid is False


def test_otel_extract_opens_span() -> None:
    _ensure_provider()
    tracer = trace.get_tracer("producer")
    payload: dict[str, Any] = {}
    with tracer.start_as_current_span("api.tasks.submit"):
        # mimic the producer-side injection
        from opentelemetry.propagate import inject

        headers: dict[str, str] = {}
        inject(headers)
        payload["traceparent"] = headers["traceparent"]

    with otel_extract(payload, "worker.run_task") as span:
        assert span.is_recording()
