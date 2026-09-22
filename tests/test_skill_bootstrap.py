from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.skill.bootstrap import get_skill_manager, reset_skill_manager
from deepresearch.skill.config import SkillConfig


@pytest.fixture(autouse=True)
def _isolated_manager():  # noqa: ANN202
    reset_skill_manager()
    yield
    reset_skill_manager()


def test_disabled_config_does_not_create_skill_storage(tmp_path: Path) -> None:
    config = SkillConfig(
        db_path=tmp_path / "skills.db",
        storage_path=tmp_path / "objects",
        include_builtin=False,
    )

    assert get_skill_manager(config) is None
    assert not (tmp_path / "skills.db").exists()
    assert not (tmp_path / "objects").exists()


def test_manager_is_reused_for_equal_process_config(tmp_path: Path) -> None:
    config = SkillConfig(
        use="default",
        db_path=tmp_path / "skills.db",
        storage_path=tmp_path / "objects",
        skill_dirs=(tmp_path / "missing",),
        include_builtin=False,
    )

    first = get_skill_manager(config)
    second = get_skill_manager(config)

    assert first is not None
    assert first is second


def test_unknown_manager_strategy_is_rejected(tmp_path: Path) -> None:
    config = SkillConfig(
        use="remote",
        db_path=tmp_path / "skills.db",
        storage_path=tmp_path / "objects",
    )

    with pytest.raises(ValueError, match="不支持的 SkillManager"):
        get_skill_manager(config)
