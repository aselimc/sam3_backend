"""Integration: S3Backend round-trip against compose MinIO."""

from __future__ import annotations

import urllib.request

import pytest

from packages.storage.s3 import S3Backend


@pytest.fixture
def backend(s3_client) -> S3Backend:
    return S3Backend(client=s3_client)


def test_put_get_head(backend: S3Backend, bucket: str) -> None:
    meta = backend.put_object(bucket, "k1", b"hello", content_type="text/plain")
    assert meta.size_bytes == 5
    assert backend.get_object(bucket, "k1") == b"hello"
    h = backend.head_object(bucket, "k1")
    assert h.content_type == "text/plain"


def test_presigned_put_then_get(backend: S3Backend, bucket: str) -> None:
    p = backend.presign_put(
        bucket, "k2", content_type="application/octet-stream", content_length=11, ttl_s=120
    )
    req = urllib.request.Request(
        p.url,
        data=b"hello world",
        method="PUT",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as r:  # noqa: S310 - test-only, signed local URL
        assert 200 <= r.status < 300
    g = backend.presign_get(bucket, "k2", ttl_s=60)
    with urllib.request.urlopen(g.url) as r:  # noqa: S310
        body = r.read()
    assert body == b"hello world"


def test_list_objects(backend: S3Backend, bucket: str) -> None:
    backend.put_object(bucket, "a/1", b"1")
    backend.put_object(bucket, "a/2", b"2")
    backend.put_object(bucket, "b/1", b"3")
    keys = sorted(o.storage_key for o in backend.list_objects(bucket, prefix="a/"))
    assert keys == ["a/1", "a/2"]
