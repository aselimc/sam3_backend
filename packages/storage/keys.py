"""Object key templates.

Layouts come from upgrade/06-storage-and-security.md §Buckets and key layout:

    uploads:   uploads/{owner}/{yyyy}/{mm}/{dd}/{uuid}
    artifacts: artifacts/{owner}/{yyyy}/{mm}/{dd}/{job_id}/{name}.{ext}
    artifact meta: artifacts/{owner}/{yyyy}/{mm}/{dd}/{job_id}/_meta.json

Owner segment is always present; in the local profile it is the constant
"local". Date prefixes amortize S3 partitioning; job_id segment scopes
artifacts so retries land in attempt=N siblings without overwrite.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import UUID

S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")


def _date_segments(when: datetime | None) -> tuple[str, str, str]:
    dt = when or datetime.now(UTC)
    return f"{dt.year:04d}", f"{dt.month:02d}", f"{dt.day:02d}"


def upload_key(owner_id: str, upload_id: UUID | str, *, when: datetime | None = None) -> str:
    y, m, d = _date_segments(when)
    return f"uploads/{owner_id}/{y}/{m}/{d}/{upload_id}"


def artifact_key(
    owner_id: str,
    job_id: UUID | str,
    name: str,
    *,
    attempt: int = 1,
    when: datetime | None = None,
) -> str:
    y, m, d = _date_segments(when)
    base = f"artifacts/{owner_id}/{y}/{m}/{d}/{job_id}"
    if attempt > 1:
        base = f"{base}/attempt={attempt}"
    return f"{base}/{name}"


def artifact_meta_key(
    owner_id: str,
    job_id: UUID | str,
    *,
    attempt: int = 1,
    when: datetime | None = None,
) -> str:
    return artifact_key(owner_id, job_id, "_meta.json", attempt=attempt, when=when)


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    m = S3_URI_RE.match(uri)
    if not m:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return m.group("bucket"), m.group("key")
