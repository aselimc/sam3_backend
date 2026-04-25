"""Storage backends + key layout helpers.

Routers and workers depend on the `StorageBackend` ABC, never on a
concrete impl. Local tests use `LocalBackend`; prod uses `S3Backend`
against MinIO (compose) or AWS S3 (enterprise).
"""

from .base import IORef, ObjectMeta, PresignedDownload, PresignedUpload, StorageBackend
from .keys import (
    artifact_key,
    artifact_meta_key,
    parse_s3_uri,
    s3_uri,
    upload_key,
)
from .local import LocalBackend
from .s3 import S3Backend, ioref_from_meta

__all__ = [
    "IORef",
    "ObjectMeta",
    "PresignedDownload",
    "PresignedUpload",
    "StorageBackend",
    "S3Backend",
    "LocalBackend",
    "ioref_from_meta",
    "upload_key",
    "artifact_key",
    "artifact_meta_key",
    "s3_uri",
    "parse_s3_uri",
]
