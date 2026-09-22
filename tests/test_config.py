import pytest

from deepresearch import config as config_module
from deepresearch.config import Settings
from deepresearch.memory.config import MemoryConfig
from deepresearch.skill.config import SkillConfig


def test_settings_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit tests must not inherit a developer's real local .env file.
    monkeypatch.setattr(config_module, "load_dotenv", lambda: False)
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


def test_memory_config_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPRESEARCH_MEMORY_USE", raising=False)

    config = MemoryConfig.from_env()

    assert config.enabled is False
    assert config.enable_recall is True
    assert config.enable_extract is False


def test_settings_read_memory_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPRESEARCH_API_KEY", "test-key")
    monkeypatch.setenv("DEEPRESEARCH_MEMORY_USE", "default")
    monkeypatch.setenv("DEEPRESEARCH_MEMORY_NAMESPACE", "project-a")
    monkeypatch.setenv("DEEPRESEARCH_MEMORY_ENABLE_EXTRACT", "true")
    monkeypatch.setenv("DEEPRESEARCH_MEMORY_TOP_K", "3")

    settings = Settings.from_env()

    assert settings.memory.enabled is True
    assert settings.memory.namespace == "project-a"
    assert settings.memory.enable_extract is True
    assert settings.memory.top_k == 3


def test_memory_config_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPRESEARCH_MEMORY_ENABLE_RECALL", "sometimes")

    with pytest.raises(ValueError, match="MEMORY_ENABLE_RECALL"):
        MemoryConfig.from_env()


def test_settings_read_skill_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPRESEARCH_API_KEY", "test-key")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_USE", "default")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_MAX_SKILLS", "2")

    settings = Settings.from_env()

    assert isinstance(settings.skill, SkillConfig)
    assert settings.skill.enabled is True
    assert settings.skill.max_skills == 2
