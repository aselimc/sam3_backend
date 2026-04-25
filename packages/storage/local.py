"""Filesystem backend used by tests.

Mirrors the StorageBackend surface but lives on disk. Presigning issues a
short-lived HMAC-signed URL pointing back at the API process — the call
site is identical to S3, but tests do not need a MinIO container.

The signing scheme is intentionally minimal: `sig = hmac_sha256(secret,
"{method}\n{bucket}\n{key}\n{exp}")`. The API mounts a verify-and-stream
endpoint in tests; production never instantiates this backend.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from packages.storage.base import (
    ObjectMeta,
    PresignedDownload,
    PresignedUpload,
    StorageBackend,
)


def _sign(secret: str, method: str, bucket: str, key: str, exp: int) -> str:
    msg = f"{method}\n{bucket}\n{key}\n{exp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_signature(
    secret: str, method: str, bucket: str, key: str, exp: int, sig: str
) -> bool:
    if exp < int(time.time()):
        return False
    expected = _sign(secret, method, bucket, key, exp)
    return hmac.compare_digest(expected, sig)


class LocalBackend(StorageBackend):
    def __init__(
        self,
        root: Path | str,
        *,
        sign_secret: str | None = None,
        base_url: str = "http://localhost:0/_local-storage",
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._secret = sign_secret or secrets.token_hex(16)
        self._base_url = base_url.rstrip("/")
        self._uploads: dict[str, dict[int, bytes]] = {}

    @property
    def root(self) -> Path:
        return self._root

    @property
    def sign_secret(self) -> str:
        return self._secret

    def _bucket_path(self, bucket: str) -> Path:
        return self._root / bucket

    def _object_path(self, bucket: str, key: str) -> Path:
        return self._bucket_path(bucket) / key

    # ── Buckets ─────────────────────────────────────────────────────────
    def ensure_bucket(self, bucket: str) -> None:
        self._bucket_path(bucket).mkdir(parents=True, exist_ok=True)

    # ── Object I/O ──────────────────────────────────────────────────────
    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
    ) -> ObjectMeta:
        p = self._object_path(bucket, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        if content_type:
            p.with_suffix(p.suffix + ".ct").write_text(content_type, encoding="utf-8")
        return self.head_object(bucket, key)

    def get_object(self, bucket: str, key: str) -> bytes:
        p = self._object_path(bucket, key)
        if not p.exists():
            raise FileNotFoundError(f"{bucket}/{key}")
        return p.read_bytes()

    def head_object(self, bucket: str, key: str) -> ObjectMeta:
        p = self._object_path(bucket, key)
        if not p.exists():
            raise FileNotFoundError(f"{bucket}/{key}")
        ct_path = p.with_suffix(p.suffix + ".ct")
        ct = ct_path.read_text(encoding="utf-8") if ct_path.exists() else None
        return ObjectMeta(
            storage_key=key,
            bucket=bucket,
            size_bytes=p.stat().st_size,
            content_type=ct,
            checksum_sha256=None,
            version_id=None,
        )

    def delete_object(self, bucket: str, key: str) -> None:
        p = self._object_path(bucket, key)
        if p.exists():
            p.unlink()
        ct = p.with_suffix(p.suffix + ".ct")
        if ct.exists():
            ct.unlink()

    def list_objects(self, bucket: str, prefix: str = "") -> Iterator[ObjectMeta]:
        base = self._bucket_path(bucket)
        if not base.exists():
            return
        for path in base.rglob("*"):
            if path.is_file() and not path.name.endswith(".ct"):
                rel = path.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    yield self.head_object(bucket, rel)

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
        exp = int(time.time()) + ttl_s
        sig = _sign(self._secret, "PUT", bucket, key, exp)
        url = f"{self._base_url}/{bucket}/{key}?exp={exp}&sig={sig}"
        return PresignedUpload(
            url=url,
            headers={"Content-Type": content_type, "Content-Length": str(content_length)},
            expires_at_epoch_s=exp,
        )

    def presign_get(
        self,
        bucket: str,
        key: str,
        *,
        ttl_s: int = 600,
        response_content_disposition: str | None = None,
    ) -> PresignedDownload:
        exp = int(time.time()) + ttl_s
        sig = _sign(self._secret, "GET", bucket, key, exp)
        url = f"{self._base_url}/{bucket}/{key}?exp={exp}&sig={sig}"
        return PresignedDownload(url=url, expires_at_epoch_s=exp)

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
        upload_id = secrets.token_hex(8)
        self._uploads[upload_id] = {}
        exp = int(time.time()) + ttl_s
        parts: list[dict[str, Any]] = []
        for i in range(1, n_parts + 1):
            sig = _sign(self._secret, f"PART:{upload_id}:{i}", bucket, key, exp)
            url = (
                f"{self._base_url}/{bucket}/{key}?upload_id={upload_id}"
                f"&part={i}&exp={exp}&sig={sig}"
            )
            parts.append({"part_number": i, "url": url})
        return PresignedUpload(
            url="",
            headers={},
            expires_at_epoch_s=exp,
            multipart={"upload_id": upload_id, "part_size": part_size, "parts": parts},
        )

    def upload_part(self, upload_id: str, part_number: int, body: bytes) -> str:
        self._uploads.setdefault(upload_id, {})[part_number] = body
        return hashlib.md5(body, usedforsecurity=False).hexdigest()

    def complete_multipart_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        parts: list[dict],
    ) -> ObjectMeta:
        chunks = self._uploads.pop(upload_id, {})
        ordered = sorted(parts, key=lambda p: int(p["part_number"]))
        body = b"".join(chunks[int(p["part_number"])] for p in ordered)
        return self.put_object(bucket, key, body)

    def abort_multipart_upload(self, bucket: str, key: str, upload_id: str) -> None:
        self._uploads.pop(upload_id, None)

    # ── Test helpers ────────────────────────────────────────────────────
    def reset(self) -> None:
        if self._root.exists():
            shutil.rmtree(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._uploads.clear()
