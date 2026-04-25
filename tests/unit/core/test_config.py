from packages.core.config import Settings, get_settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.app_env == "local"
    assert s.max_upload_bytes == 50 * 1024 * 1024
    assert s.max_image_pixels > 0
    assert s.s3_bucket_uploads == "sam3-uploads"
    assert "sam3" in s.models_enabled


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "12345")
    monkeypatch.setenv("APP_ENV", "test")
    s = Settings(_env_file=None)
    assert s.max_upload_bytes == 12345
    assert s.app_env == "test"


def test_get_settings_singleton():
    a = get_settings()
    b = get_settings()
    assert a is b
