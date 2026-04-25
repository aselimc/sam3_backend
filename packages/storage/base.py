"""Storage backend ABC.

`StorageBackend` is the contract every concrete impl honors (S3/MinIO and
local FS). Routers + workers depend on the ABC, never on a concrete class —
local tests swap to `LocalBackend` and prod uses `S3Backend`.

`IORef` is the lightweight pointer carried inside typed I/O classes. It
matches the JSON shape the API accepts in upload responses and request
payloads (see upgrade/03-api-spec.md §Uploads).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from packages.core.schemas import VersionedModel


class IORef(VersionedModel):
    """Pointer to an object in storage; embedded inside InputBase fields."""

    storage_key: Annotated[str, Field(min_length=1)]
    content_type: str | None = None
    byte_length: int | None = Field(default=None, ge=0)
    checksum_sha256: str | None = None
    version_id: str | None = None


@dataclass(slots=True)
class PresignedUpload:
    url: str
    headers: dict[str, str]
    expires_at_epoch_s: int
    multipart: dict | None = None  # populated for multipart starts


@dataclass(slots=True)
class PresignedDownload:
    url: str
    expires_at_epoch_s: int


@dataclass(slots=True)
class ObjectMeta:
    storage_key: str
    bucket: str
    size_bytes: int
    content_type: str | None
    checksum_sha256: str | None
    version_id: str | None


class StorageBackend(ABC):
    """Common surface for S3-compatible and local-filesystem backends."""

    upload_method: Literal["PUT"] = "PUT"

    # ── Buckets ─────────────────────────────────────────────────────────
    @abstractmethod
    def ensure_bucket(self, bucket: str) -> None: ...

    # ── Object I/O ──────────────────────────────────────────────────────
    @abstractmethod
    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
    ) -> ObjectMeta: ...

    @abstractmethod
    def get_object(self, bucket: str, key: str) -> bytes: ...

    @abstractmethod
    def head_object(self, bucket: str, key: str) -> ObjectMeta: ...

    @abstractmethod
    def delete_object(self, bucket: str, key: str) -> None: ...

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> Iterator[ObjectMeta]: ...

    # ── Presigning ──────────────────────────────────────────────────────
    @abstractmethod
    def presign_put(
        self,
        bucket: str,
        key: str,
        *,
        content_type: str,
        content_length: int,
        ttl_s: int = 900,
    ) -> PresignedUpload: ...

    @abstractmethod
    def presign_get(
        self,
        bucket: str,
        key: str,
        *,
        ttl_s: int = 600,
        response_content_disposition: str | None = None,
    ) -> PresignedDownload: ...

    # ── Multipart ───────────────────────────────────────────────────────
    @abstractmethod
    def create_multipart_upload(
        self,
        bucket: str,
        key: str,
        *,
        content_type: str,
        n_parts: int,
        part_size: int,
        ttl_s: int = 3600,
    ) -> PresignedUpload: ...

    @abstractmethod
    def complete_multipart_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        parts: list[dict],
    ) -> ObjectMeta: ...

    @abstractmethod
    def abort_multipart_upload(self, bucket: str, key: str, upload_id: str) -> None: ...
