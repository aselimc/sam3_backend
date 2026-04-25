"""100% coverage on errors.py."""

import pytest

from packages.core import errors as e


@pytest.mark.parametrize(
    "cls,code,status",
    [
        (e.Unauthorized, "unauthorized", 401),
        (e.Forbidden, "forbidden", 403),
        (e.NotFound, "not_found", 404),
        (e.ValidationError, "validation_error", 422),
        (e.IdempotencyRequired, "idempotency_required", 400),
        (e.IdempotencyConflict, "idempotency_conflict", 409),
        (e.StateConflict, "state_conflict", 409),
        (e.RateLimited, "rate_limited", 429),
        (e.PayloadTooLarge, "payload_too_large", 413),
        (e.UnsupportedMediaType, "unsupported_media_type", 415),
        (e.ModelUnavailable, "model_unavailable", 503),
        (e.Internal, "internal", 500),
        (e.TransientError, "transient_io", 503),
        (e.PreflightOOM, "preflight_oom", 503),
        (e.RuntimeOOM, "runtime_oom", 503),
        (e.CancelRequested, "canceled", 499),
        (e.AdapterError, "adapter_error", 500),
    ],
)
def test_class_has_code_and_status(cls, code, status):
    assert cls.code == code
    assert cls.http_status == status
    inst = cls()
    assert inst.message == code
    assert isinstance(inst, e.AppError)
    assert isinstance(inst, Exception)


def test_envelope_with_details():
    err = e.RateLimited("Too many requests", retry_after_s=12)
    env = err.to_envelope()
    assert env == {
        "code": "rate_limited",
        "message": "Too many requests",
        "details": {"retry_after_s": 12},
    }


def test_envelope_without_details():
    err = e.NotFound()
    env = err.to_envelope()
    assert env == {"code": "not_found", "message": "not_found"}


def test_by_code_known():
    assert e.by_code("rate_limited") is e.RateLimited
    assert e.by_code("unauthorized") is e.Unauthorized


def test_by_code_unknown_falls_back_to_internal():
    assert e.by_code("definitely-not-real") is e.Internal


def test_app_error_default_code_is_internal():
    err = e.AppError("boom", x=1)
    assert err.code == "internal"
    assert err.http_status == 500
    assert err.details == {"x": 1}


def test_app_error_is_exception_with_message():
    err = e.ValidationError("bad payload")
    assert str(err) == "bad payload"
