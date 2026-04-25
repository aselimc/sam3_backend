"""LocalBackend round-trip tests."""

from __future__ import annotations

import pytest

from packages.storage.local import LocalBackend, verify_signature


@pytest.fixture
def backend(tmp_path) -> LocalBackend:
    b = LocalBackend(tmp_path / "store", sign_secret="s3cret")
    b.ensure_bucket("uploads")
    b.ensure_bucket("artifacts")
    return b


def test_put_get_roundtrip(backend: LocalBackend) -> None:
    meta = backend.put_object("uploads", "k1", b"hello", content_type="text/plain")
    assert meta.size_bytes == 5
    assert meta.content_type == "text/plain"
    assert backend.get_object("uploads", "k1") == b"hello"


def test_head_after_put(backend: LocalBackend) -> None:
    backend.put_object("uploads", "k2", b"world")
    h = backend.head_object("uploads", "k2")
    assert h.bucket == "uploads"
    assert h.size_bytes == 5


def test_delete_object(backend: LocalBackend) -> None:
    backend.put_object("uploads", "k3", b"x")
    backend.delete_object("uploads", "k3")
    with pytest.raises(FileNotFoundError):
        backend.get_object("uploads", "k3")


def test_list_objects(backend: LocalBackend) -> None:
    backend.put_object("uploads", "a/1", b"1")
    backend.put_object("uploads", "a/2", b"2")
    backend.put_object("uploads", "b/3", b"3")
    keys = sorted(o.storage_key for o in backend.list_objects("uploads", prefix="a/"))
    assert keys == ["a/1", "a/2"]


def test_presign_put_returns_signed_url(backend: LocalBackend) -> None:
    p = backend.presign_put("uploads", "k", content_type="image/png", content_length=10, ttl_s=60)
    assert "sig=" in p.url and "exp=" in p.url
    assert p.headers == {"Content-Type": "image/png", "Content-Length": "10"}
    # signature verifies
    exp = int(p.url.split("exp=")[1].split("&")[0])
    sig = p.url.split("sig=")[1]
    assert verify_signature(backend.sign_secret, "PUT", "uploads", "k", exp, sig)


def test_presign_get(backend: LocalBackend) -> None:
    p = backend.presign_get("artifacts", "out.png", ttl_s=60)
    assert "sig=" in p.url
    assert p.expires_at_epoch_s > 0


def test_multipart_roundtrip(backend: LocalBackend) -> None:
    init = backend.create_multipart_upload(
        "uploads", "big.bin", content_type="application/octet-stream", n_parts=2, part_size=4
    )
    upload_id = init.multipart["upload_id"]
    e1 = backend.upload_part(upload_id, 1, b"1234")
    e2 = backend.upload_part(upload_id, 2, b"5678")
    meta = backend.complete_multipart_upload(
        "uploads",
        "big.bin",
        upload_id,
        [{"part_number": 1, "etag": e1}, {"part_number": 2, "etag": e2}],
    )
    assert meta.size_bytes == 8
    assert backend.get_object("uploads", "big.bin") == b"12345678"
