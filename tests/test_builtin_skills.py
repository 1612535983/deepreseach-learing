from __future__ import annotations

from pathlib import Path

from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import BUILTIN_SKILLS_DIR, SkillManager
from deepresearch.skill.parser import discover_skills


def test_builtin_skill_bundles_are_valid_and_complete() -> None:
    discovered = discover_skills([BUILTIN_SKILLS_DIR], origin="BUILTIN")

    assert discovered.errors == ()
    assert {item.record.name for item in discovered.skills} == {
        "web-research",
        "source-verification",
        "evidence-report-writing",
    }
    assert all(item.record.lineage.origin == "BUILTIN" for item in discovered.skills)


def test_manager_loads_builtins_and_user_bundle_can_override_name(
    tmp_path: Path,
) -> None:
    override = tmp_path / "skills" / "source-verification"
    override.mkdir(parents=True)
    (override / "SKILL.md").write_text(
        """---
name: source-verification
description: 项目专用来源核验流程
version: 2
tags: [核验]
tools: [read_page]
---
只接受本项目登记的一手数据源。
""",
        encoding="utf-8",
    )
    manager = SkillManager(
        SkillConfig(
            use="default",
            db_path=tmp_path / "skills.db",
            storage_path=tmp_path / "objects",
            skill_dirs=(tmp_path / "skills",),
            include_builtin=True,
        )
    )

    manager.load_startup()

    active = {record.name: record for record in manager.list_skills()}
    assert set(active) == {
        "web-research",
        "source-verification",
        "evidence-report-writing",
    }
    assert active["source-verification"].version == 2
    assert active["source-verification"].lineage.origin == "IMPORTED"
    assert len(manager.store.get_versions("source-verification")) == 2
    manager.close()
