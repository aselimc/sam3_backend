"""S3Backend presign tests using a stubbed boto3 client."""

from __future__ import annotations

from unittest.mock import MagicMock

from packages.storage.s3 import S3Backend


def _make_backend() -> tuple[S3Backend, MagicMock]:
    client = MagicMock()
    client.generate_presigned_url.return_value = "https://example/signed"
    backend = S3Backend(client=client)
    return backend, client


def test_presign_put_passes_content_headers() -> None:
    backend, client = _make_backend()
    p = backend.presign_put(
        "uploads", "k", content_type="image/png", content_length=42, ttl_s=300
    )
    assert p.url == "https://example/signed"
    assert p.headers == {"Content-Type": "image/png", "Content-Length": "42"}
    args = client.generate_presigned_url.call_args
    assert args.kwargs["ClientMethod"] == "put_object"
    assert args.kwargs["Params"]["ContentType"] == "image/png"
    assert args.kwargs["Params"]["ContentLength"] == 42
    assert args.kwargs["ExpiresIn"] == 300


def test_presign_get_disposition_passed() -> None:
    backend, client = _make_backend()
    disp = "attachment; filename=x.png"
    backend.presign_get("artifacts", "k", ttl_s=600, response_content_disposition=disp)
    args = client.generate_presigned_url.call_args
    assert args.kwargs["Params"]["ResponseContentDisposition"] == disp


def test_create_multipart_returns_per_part_urls() -> None:
    backend, client = _make_backend()
    client.create_multipart_upload.return_value = {"UploadId": "uid-1"}
    init = backend.create_multipart_upload(
        "uploads", "k", content_type="image/png", n_parts=3, part_size=16
    )
    assert init.multipart["upload_id"] == "uid-1"
    assert len(init.multipart["parts"]) == 3
    assert all(p["url"] == "https://example/signed" for p in init.multipart["parts"])


def test_complete_multipart_maps_etags() -> None:
    backend, client = _make_backend()
    client.head_object.return_value = {
        "ContentLength": 10,
        "ContentType": "image/png",
    }
    backend.complete_multipart_upload(
        "uploads",
        "k",
        "uid",
        [{"part_number": 1, "etag": "e1"}, {"part_number": 2, "etag": "e2"}],
    )
    call = client.complete_multipart_upload.call_args.kwargs
    assert call["MultipartUpload"]["Parts"] == [
        {"PartNumber": 1, "ETag": "e1"},
        {"PartNumber": 2, "ETag": "e2"},
    ]
