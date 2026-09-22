from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.skill.exceptions import SkillValidationError
from deepresearch.skill.parser import discover_skills, parse_skill_file


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "验证来源",
    extra: str = "",
    body: str = "# Source Verification\n\n交叉核验来源。",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "version: 2",
                "tags: [research, verification, research]",
                "tools: [web_search, read_page]",
                "enabled: true",
                extra,
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_parse_skill_file_builds_deterministic_content_version(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "source-verification")

    first = parse_skill_file(path, origin="BUILTIN")
    second = parse_skill_file(path, origin="BUILTIN")

    assert first.record.skill_id == second.record.skill_id
    assert first.record.skill_id.startswith("source-verification__")
    assert len(first.record.content_hash) == 64
    assert first.record.version == 2
    assert first.record.tags == ("research", "verification")
    assert first.record.allowed_tools == ("web_search", "read_page")
    assert first.record.lineage.origin == "BUILTIN"
    assert first.body.startswith("# Source Verification")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# no metadata", "frontmatter"),
        ("---\ndescription: missing name\n---\nbody", "name"),
        ("---\nname: Bad_Name\ndescription: bad\n---\nbody", "kebab-case"),
        ("---\nname: valid\ndescription: ''\n---\nbody", "description"),
        ("---\nname: valid\ndescription: ok\nversion: 0\n---\nbody", "version"),
        ("---\nname: valid\ndescription: ok\ntags: tag\n---\nbody", "tags"),
        ("---\nname: valid\ndescription: ok\n---\n", "正文"),
    ],
)
def test_parse_skill_file_rejects_invalid_contract(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SkillValidationError, match=message):
        parse_skill_file(path)


def test_parse_skill_file_accepts_allowed_tools_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(
        "---\nname: verify\ndescription: Verify\nallowed-tools:\n  - read_page\n---\nbody",
        encoding="utf-8",
    )

    parsed = parse_skill_file(path)

    assert parsed.record.allowed_tools == ("read_page",)


def test_parse_skill_file_enforces_size_limit(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "verify")

    with pytest.raises(SkillValidationError, match="超过限制"):
        parse_skill_file(path, max_file_chars=20)


def test_discovery_is_sorted_and_isolates_bad_bundles(tmp_path: Path) -> None:
    _write_skill(tmp_path, "zeta")
    _write_skill(tmp_path, "alpha")
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_text("broken", encoding="utf-8")

    result = discover_skills([tmp_path])

    assert [skill.record.name for skill in result.skills] == ["alpha", "zeta"]
    assert len(result.errors) == 1
    assert result.errors[0].path.endswith("broken/SKILL.md")


def test_discovery_ignores_missing_directory(tmp_path: Path) -> None:
    result = discover_skills([tmp_path / "missing"])

    assert result.skills == ()
    assert result.errors == ()
