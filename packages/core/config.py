"""Env-driven settings (pydantic-settings).

Reads `.env` at the repo root. Routers, workers, scripts all use the same
Settings() — divergence is a bug. Defaults match upgrade/03 + 06 + 08.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ─────────────────────────────────────────────────────────────
    app_env: str = "local"
    log_level: str = "INFO"
    service_name: str = "api"  # set per-process: "api", "worker", "beat", "webhook"
    version: str = "0.0.0+dev"

    # ── Auth (local profile) ────────────────────────────────────────────
    local_api_key: str = "replace-me-with-random-32-byte-hex"
    webhook_secret: str = "replace-me-with-random-32-byte-hex"

    # ── API ─────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_bytes: int = 50 * 1024 * 1024
    max_image_pixels: int = 178_956_970  # PIL default Decompression-bomb threshold

    # ── Postgres ────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://sam3:sam3@postgres:5432/sam3"

    # ── Redis / Celery ──────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # ── S3 / MinIO ──────────────────────────────────────────────────────
    s3_endpoint_url: str = "http://minio:9000"
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_uploads: str = "sam3-uploads"
    s3_bucket_artifacts: str = "sam3-artifacts"
    s3_force_path_style: bool = True

    # ── Worker ──────────────────────────────────────────────────────────
    models_enabled: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["sam3", "depth_anything_v3"]
    )
    worker_preload: bool = False
    cuda_visible_devices: str = "0"

    @field_validator("models_enabled", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # ── Telemetry ───────────────────────────────────────────────────────
    log_prompts: bool = False
    metrics_port: int = 9100
    otel_endpoint: str | None = None  # OTLP collector; None = no exporter


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
