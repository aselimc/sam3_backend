"""AppError hierarchy with stable error codes and HTTP mappings.

Every code in upgrade/03-api-spec.md §Errors has a class. Routers raise
these; an error-handler middleware (Phase 6) renders the single envelope:

    { "error": { "code": ..., "message": ..., "details": {...} } }
"""

from __future__ import annotations

from typing import Any, ClassVar


class AppError(Exception):
    code: ClassVar[str] = "internal"
    http_status: ClassVar[int] = 500
    message: str
    details: dict[str, Any]

    def __init__(self, message: str | None = None, **details: Any) -> None:
        self.message = message or self.code
        self.details = details
        super().__init__(self.message)

    def to_envelope(self) -> dict[str, Any]:
        env: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            env["details"] = self.details
        return env


class Unauthorized(AppError):
    code = "unauthorized"
    http_status = 401


class Forbidden(AppError):
    code = "forbidden"
    http_status = 403


class NotFound(AppError):
    code = "not_found"
    http_status = 404


class ValidationError(AppError):
    code = "validation_error"
    http_status = 422


class IdempotencyRequired(AppError):
    code = "idempotency_required"
    http_status = 400


class IdempotencyConflict(AppError):
    code = "idempotency_conflict"
    http_status = 409


class StateConflict(AppError):
    code = "state_conflict"
    http_status = 409


class RateLimited(AppError):
    code = "rate_limited"
    http_status = 429


class PayloadTooLarge(AppError):
    code = "payload_too_large"
    http_status = 413


class UnsupportedMediaType(AppError):
    code = "unsupported_media_type"
    http_status = 415


class ModelUnavailable(AppError):
    code = "model_unavailable"
    http_status = 503


class Internal(AppError):
    code = "internal"
    http_status = 500


# Worker-internal error taxonomy (07/05). Not surfaced as HTTP but flow into
# JobRecord.error_code via fmt_error in the runner.

class TransientError(AppError):
    """HF/S3 transient I/O — auto-retried."""

    code = "transient_io"
    http_status = 503


class PreflightOOM(AppError):
    """GPU mem probe rejected the task before it ran. Retry without attempt cost."""

    code = "preflight_oom"
    http_status = 503


class RuntimeOOM(AppError):
    """torch.cuda.OutOfMemoryError caught by oom_guard. Bump gpu_class on retry."""

    code = "runtime_oom"
    http_status = 503


class CancelRequested(AppError):
    """SIGUSR1 raised by CancelCheck inside a long loop. Worker frees GPU and exits cleanly."""

    code = "canceled"
    http_status = 499


class AdapterError(AppError):
    """Adapter raised something we did not expect. No retry."""

    code = "adapter_error"
    http_status = 500


# Code → class lookup, useful for fmt_error and tests.
_BY_CODE: dict[str, type[AppError]] = {
    cls.code: cls
    for cls in (
        Unauthorized,
        Forbidden,
        NotFound,
        ValidationError,
        IdempotencyRequired,
        IdempotencyConflict,
        StateConflict,
        RateLimited,
        PayloadTooLarge,
        UnsupportedMediaType,
        ModelUnavailable,
        Internal,
        TransientError,
        PreflightOOM,
        RuntimeOOM,
        CancelRequested,
        AdapterError,
    )
}


def by_code(code: str) -> type[AppError]:
    return _BY_CODE.get(code, Internal)
