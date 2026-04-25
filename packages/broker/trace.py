"""W3C trace-context propagation across the Celery payload.

OTel instrumentation does not auto-propagate across Redis-broker Celery
hops, so we inject `traceparent` / `tracestate` into the payload on
submit, and extract on the worker side. See upgrade/05-worker-runtime.md
§Tracing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from celery import Celery
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject

from packages.core.telemetry import get_tracer


def submit_with_trace(
    celery_app: Celery,
    task_name: str,
    payload: dict[str, Any],
    *,
    queue: str | None = None,
    **send_kwargs: Any,
) -> Any:
    """Inject the active trace context into `payload` and send the task.

    Mutates `payload` to add `traceparent` / `tracestate` keys; callers
    pass the same dict downstream so the worker recovers the context.
    """
    headers: dict[str, str] = {}
    inject(headers)
    if (tp := headers.get("traceparent")) is not None:
        payload["traceparent"] = tp
    if (ts := headers.get("tracestate")) is not None:
        payload["tracestate"] = ts
    return celery_app.send_task(task_name, args=[payload], queue=queue, **send_kwargs)


def extract_from_payload(payload: dict[str, Any]) -> Context:
    carrier = {
        "traceparent": payload.get("traceparent", "") or "",
        "tracestate": payload.get("tracestate", "") or "",
    }
    return extract(carrier)


@contextmanager
def otel_extract(
    payload: dict[str, Any],
    span_name: str = "worker.run_task",
) -> Iterator[trace.Span]:
    """Open a server span using the trace context carried in the payload."""
    ctx = extract_from_payload(payload)
    tracer = get_tracer()
    with tracer.start_as_current_span(span_name, context=ctx) as span:
        yield span
