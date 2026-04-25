"""Unit tests for storage key templates."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from packages.storage.keys import (
    artifact_key,
    artifact_meta_key,
    parse_s3_uri,
    s3_uri,
    upload_key,
)

WHEN = datetime(2026, 4, 25, 10, 0, 0, tzinfo=UTC)
UPLOAD_ID = UUID("00000000-0000-0000-0000-000000000abc")
JOB_ID = UUID("11111111-1111-1111-1111-111111111111")


def test_upload_key_layout() -> None:
    assert upload_key("local", UPLOAD_ID, when=WHEN) == (
        "uploads/local/2026/04/25/00000000-0000-0000-0000-000000000abc"
    )


def test_artifact_key_first_attempt_no_attempt_segment() -> None:
    assert (
        artifact_key("local", JOB_ID, "depth.png", when=WHEN)
        == "artifacts/local/2026/04/25/11111111-1111-1111-1111-111111111111/depth.png"
    )


def test_artifact_key_retry_uses_attempt_sibling() -> None:
    assert (
        artifact_key("local", JOB_ID, "depth.png", attempt=2, when=WHEN)
        == "artifacts/local/2026/04/25/11111111-1111-1111-1111-111111111111/attempt=2/depth.png"
    )


def test_artifact_meta_key() -> None:
    assert artifact_meta_key("local", JOB_ID, when=WHEN).endswith("/_meta.json")


def test_s3_uri_roundtrip() -> None:
    uri = s3_uri("sam3-uploads", "uploads/local/abc")
    assert uri == "s3://sam3-uploads/uploads/local/abc"
    assert parse_s3_uri(uri) == ("sam3-uploads", "uploads/local/abc")


def test_parse_s3_uri_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_s3_uri("not-a-uri")
