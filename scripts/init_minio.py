"""Idempotent MinIO bootstrapping.

Creates the two buckets (`uploads` + `artifacts`) declared in
upgrade/06 §Buckets and key layout. Safe to re-run; existing buckets
are left alone.
"""

from __future__ import annotations

import sys

from packages.core.config import get_settings
from packages.storage.s3 import S3Backend


def main() -> int:
    s = get_settings()
    backend = S3Backend(s)
    for bucket in (s.s3_bucket_uploads, s.s3_bucket_artifacts):
        backend.ensure_bucket(bucket)
        print(f"ok bucket={bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
