"""Integration fixtures for storage against the compose MinIO.

Skips the whole module when no S3 is reachable. Bucket names are unique
per run so parallel pytest invocations cannot collide.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import boto3
import pytest
from botocore.client import Config
from botocore.exceptions import EndpointConnectionError

S3_TEST_ENDPOINT = os.environ.get("S3_TEST_ENDPOINT", "http://localhost:9000")
S3_TEST_KEY = os.environ.get("S3_TEST_ACCESS_KEY", "minioadmin")
S3_TEST_SECRET = os.environ.get("S3_TEST_SECRET_KEY", "minioadmin")
S3_TEST_REGION = os.environ.get("S3_TEST_REGION", "us-east-1")


def _ping() -> bool:
    try:
        c = boto3.client(
            "s3",
            endpoint_url=S3_TEST_ENDPOINT,
            aws_access_key_id=S3_TEST_KEY,
            aws_secret_access_key=S3_TEST_SECRET,
            region_name=S3_TEST_REGION,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        c.list_buckets()
        return True
    except EndpointConnectionError:
        return False
    except Exception:
        return False


_HAS_S3 = _ping()


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_s3() -> None:
    if not _HAS_S3:
        pytest.skip(f"compose MinIO unreachable at {S3_TEST_ENDPOINT}", allow_module_level=True)


@pytest.fixture
def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_TEST_ENDPOINT,
        aws_access_key_id=S3_TEST_KEY,
        aws_secret_access_key=S3_TEST_SECRET,
        region_name=S3_TEST_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


@pytest.fixture
def bucket(s3_client) -> Iterator[str]:
    name = f"sam3-it-{uuid.uuid4().hex[:8]}"
    s3_client.create_bucket(Bucket=name)
    try:
        yield name
    finally:
        # empty + delete
        objs = s3_client.list_objects_v2(Bucket=name).get("Contents", []) or []
        for o in objs:
            s3_client.delete_object(Bucket=name, Key=o["Key"])
        s3_client.delete_bucket(Bucket=name)
