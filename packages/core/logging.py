"""JSON-line structured logging matching upgrade/07-observability.md §Log schema.

Every line carries the cross-tier fields needed to correlate API → broker
→ worker without parsing message text. ContextVars propagate ids across
async boundaries (FastAPI middleware, Celery task entry, beat scheduler).

Usage:

    from packages.core import logging as log
    log.configure()
    with log.bind(request_id="abc", job_id="def"):
        log.logger.info("inference done", duration_ms=14123)
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

from loguru import logger

from .config import get_settings

# ── ContextVars (one per top-level log field that varies per request/job) ──
_ctx_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_ctx_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_ctx_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)
_ctx_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_ctx_owner_id: ContextVar[str | None] = ContextVar("owner_id", default=None)
_ctx_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
_ctx_model_id: ContextVar[str | None] = ContextVar("model_id", default=None)
_ctx_gpu_class: ContextVar[str | None] = ContextVar("gpu_class", default=None)

_CTX_FIELDS: dict[str, ContextVar[str | None]] = {
    "request_id": _ctx_request_id,
    "trace_id": _ctx_trace_id,
    "span_id": _ctx_span_id,
    "tenant_id": _ctx_tenant_id,
    "owner_id": _ctx_owner_id,
    "job_id": _ctx_job_id,
    "model_id": _ctx_model_id,
    "gpu_class": _ctx_gpu_class,
}


@contextmanager
def bind(**fields: str | None) -> Iterator[None]:
    """Push values onto the relevant ContextVars for the duration of the block."""
    tokens = []
    for name, value in fields.items():
        var = _CTX_FIELDS.get(name)
        if var is None:
            continue
        tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _format(record: dict[str, Any]) -> str:
    settings = get_settings()
    out: dict[str, Any] = {
        "ts": datetime.fromtimestamp(record["time"].timestamp(), tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "level": record["level"].name,
        "service": settings.service_name,
        "version": settings.version,
        "msg": record["message"],
    }
    for name, var in _CTX_FIELDS.items():
        v = var.get()
        if v is not None:
            out[name] = v

    # bound extras (loguru "extra" dict) — flatten on top
    extras = record.get("extra") or {}
    for k, v in extras.items():
        if k in out:
            continue
        out[k] = v

    if record["exception"] is not None:
        exc = record["exception"]
        out["error.summary"] = f"{exc.type.__name__}: {exc.value}"

    return json.dumps(out, default=str, ensure_ascii=False)


def _sink(message: Any) -> None:
    record = message.record
    sys.stderr.write(_format(record) + "\n")


_configured = False


def configure(level: str | None = None) -> None:
    """Replace loguru handlers with the JSON sink. Idempotent."""
    global _configured
    settings = get_settings()
    logger.remove()
    logger.add(_sink, level=(level or settings.log_level).upper(), backtrace=False, diagnose=False)
    _configured = True


__all__ = ["bind", "configure", "logger"]
