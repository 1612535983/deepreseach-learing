import pytest

from deepresearch.config import Settings


def test_settings_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPRESEARCH_API_KEY", raising=False)
    monkeypatch.setenv("DEEPRESEARCH_BASE_URL", "https://example.com/v1")

    with pytest.raises(ValueError, match="DEEPRESEARCH_API_KEY"):
        Settings.from_env()


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPRESEARCH_API_KEY", "test-key")
    monkeypatch.setenv("DEEPRESEARCH_MODEL", "test-model")
    monkeypatch.setenv("DEEPRESEARCH_BASE_URL", "https://example.com/v1")

    settings = Settings.from_env()

    assert settings.api_key == "test-key"
    assert settings.model == "test-model"
    assert settings.base_url == "https://example.com/v1"

