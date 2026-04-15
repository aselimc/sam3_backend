from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAM3_", env_file=".env", extra="ignore")

    max_concurrent_inferences: int = 1
    log_level: str = "INFO"
    enable_metrics: bool = True


settings = Settings()
