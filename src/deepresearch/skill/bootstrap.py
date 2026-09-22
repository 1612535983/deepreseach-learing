"""Thread-safe lifecycle for the optional process-wide skill manager."""

from __future__ import annotations

import threading

from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import SkillManager


_manager_lock = threading.Lock()
_skill_manager: SkillManager | None = None
_skill_config: SkillConfig | None = None


def get_skill_manager(config: SkillConfig) -> SkillManager | None:
    """Construct and load the configured manager; an empty ``use`` disables it."""

    global _skill_manager, _skill_config
    if not config.enabled:
        return None
    if config.use != "default":
        raise ValueError(f"不支持的 SkillManager：{config.use}")
    with _manager_lock:
        if _skill_manager is not None and _skill_config == config:
            return _skill_manager
        if _skill_manager is not None:
            _skill_manager.close()
        manager = SkillManager(config)
        manager.load_startup()
        _skill_manager = manager
        _skill_config = config
        return manager


def set_skill_manager(manager: SkillManager | None) -> None:
    """Inject or clear a manager, primarily for isolated tests."""

    global _skill_manager, _skill_config
    with _manager_lock:
        if _skill_manager is not None and _skill_manager is not manager:
            _skill_manager.close()
        _skill_manager = manager
        _skill_config = manager.config if manager is not None else None


def reset_skill_manager() -> None:
    set_skill_manager(None)
