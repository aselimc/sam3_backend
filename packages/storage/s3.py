"""S3-compatible backend (boto3) — used for both MinIO and AWS S3.

Configured via Settings: endpoint, region, keys, path-style toggle. The
backend exposes the StorageBackend surface; presigned PUTs require the
client to send `Content-Type` and `Content-Length` (per
upgrade/06 §Presigning). Multipart starts return per-part presigned URLs.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from packages.core.config import Settings, get_settings
from packages.storage.base import (
    IORef,
    ObjectMeta,
    PresignedDownload,
    PresignedUpload,
    StorageBackend,
)


class S3Backend(StorageBackend):
    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        s = settings or get_settings()
        self._settings = s
        if client is not None:
            self._s3 = client
        else:
            cfg = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if s.s3_force_path_style else "auto"},
                retries={"max_attempts": 3, "mode": "standard"},
            )
            self._s3 = boto3.client(
                "s3",
                endpoint_url=s.s3_endpoint_url,
                region_name=s.s3_region,
                aws_access_key_id=s.s3_access_key,
                aws_secret_access_key=s.s3_secret_key,
                config=cfg,
            )

    @property
    def client(self) -> Any:
        return self._s3

    # ── Buckets ─────────────────────────────────────────────────────────
    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket", "NotFound"):
                self._s3.create_bucket(Bucket=bucket)
            else:
                raise

    # ── Object I/O ──────────────────────────────────────────────────────
    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
    ) -> ObjectMeta:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": body}
        if content_type:
            kwargs["ContentType"] = content_type
        self._s3.put_object(**kwargs)
        return self.head_object(bucket, key)

    def get_object(self, bucket: str, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def head_object(self, bucket: str, key: str) -> ObjectMeta:
        h = self._s3.head_object(Bucket=bucket, Key=key)
        return ObjectMeta(
            storage_key=key,
            bucket=bucket,
            size_bytes=int(h.get("ContentLength", 0)),
            content_type=h.get("ContentType"),
            checksum_sha256=h.get("ChecksumSHA256"),
            version_id=h.get("VersionId"),
        )

    def delete_object(self, bucket: str, key: str) -> None:
        self._s3.delete_object(Bucket=bucket, Key=key)

    def list_objects(self, bucket: str, prefix: str = "") -> Iterator[ObjectMeta]:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for it in page.get("Contents", []) or []:
                yield ObjectMeta(
                    storage_key=it["Key"],
                    bucket=bucket,
                    size_bytes=int(it.get("Size", 0)),
                    content_type=None,
                    checksum_sha256=None,
                    version_id=None,
                )

    # ── Presigning ──────────────────────────────────────────────────────
    def presign_put(
        self,
        bucket: str,
        key: str,
        *,
        content_type: str,
        content_length: int,
        ttl_s: int = 900,
    ) -> PresignedUpload:
        url = self._s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": content_type,
                "ContentLength": content_length,
            },
            ExpiresIn=ttl_s,
            HttpMethod="PUT",
        )
        return PresignedUpload(
            url=url,
            headers={"Content-Type": content_type, "Content-Length": str(content_length)},
            expires_at_epoch_s=int(time.time()) + ttl_s,
        )

    def presign_get(
        self,
        bucket: str,
        key: str,
        *,
        ttl_s: int = 600,
        response_content_disposition: str | None = None,
    ) -> PresignedDownload:
        params: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if response_content_disposition:
            params["ResponseContentDisposition"] = response_content_disposition
        url = self._s3.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=ttl_s,
            HttpMethod="GET",
        )
        return PresignedDownload(url=url, expires_at_epoch_s=int(time.time()) + ttl_s)

    # ── Multipart ───────────────────────────────────────────────────────
    def create_multipart_upload(
        self,
        bucket: str,
        key: str,
        *,
        content_type: str,
        n_parts: int,
        part_size: int,
        ttl_s: int = 3600,
    ) -> PresignedUpload:
        init = self._s3.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType=content_type,
        )
        upload_id = init["UploadId"]
        parts = []
        for i in range(1, n_parts + 1):
            url = self._s3.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": i,
                },
                ExpiresIn=ttl_s,
                HttpMethod="PUT",
            )
            parts.append({"part_number": i, "url": url})
        return PresignedUpload(
            url="",
            headers={},
            expires_at_epoch_s=int(time.time()) + ttl_s,
            multipart={"upload_id": upload_id, "part_size": part_size, "parts": parts},
        )

    def complete_multipart_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        parts: list[dict],
    ) -> ObjectMeta:
        self._s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={
                "Parts": [
                    {"PartNumber": int(p["part_number"]), "ETag": p["etag"]} for p in parts
                ]
            },
        )
        return self.head_object(bucket, key)

    def abort_multipart_upload(self, bucket: str, key: str, upload_id: str) -> None:
        self._s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)


def ioref_from_meta(bucket: str, meta: ObjectMeta) -> IORef:
    return IORef(
        storage_key=f"s3://{bucket}/{meta.storage_key}",
        content_type=meta.content_type,
        byte_length=meta.size_bytes,
        checksum_sha256=meta.checksum_sha256,
        version_id=meta.version_id,
    )
