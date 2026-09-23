"""Parse and safely discover Markdown skill bundles."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from deepresearch.skill.exceptions import SkillValidationError
from deepresearch.skill.types import (
    ParsedSkill,
    SkillDiscoveryError,
    SkillDiscoveryResult,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
)


_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n(.*)\Z", re.DOTALL)
_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _string_list(value: object, *, field: str, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SkillValidationError(f"{path}: {field} 必须是字符串列表。")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SkillValidationError(f"{path}: {field} 必须是非空字符串列表。")
        text = item.strip()
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _frontmatter(content: str, path: Path) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        raise SkillValidationError(
            f"{path}: 缺少 YAML frontmatter，文件必须以 --- 开始和结束。"
        )
    raw_yaml, body = match.groups()
    try:
        metadata = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        raise SkillValidationError(f"{path}: YAML 解析失败：{exc}") from exc
    if not isinstance(metadata, dict):
        raise SkillValidationError(f"{path}: YAML frontmatter 必须是 mapping。")
    if not body.strip():
        raise SkillValidationError(f"{path}: Skill 正文不能为空。")
    return metadata, body.strip()


def parse_skill_file(
    skill_file: str | Path,
    *,
    origin: SkillOrigin = "IMPORTED",
    max_file_chars: int = 50_000,
) -> ParsedSkill:
    """Validate one ``SKILL.md`` and return immutable metadata plus content."""

    path = Path(skill_file)
    if path.name != "SKILL.md":
        raise SkillValidationError(f"{path}: Skill 入口文件必须命名为 SKILL.md。")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SkillValidationError(f"{path}: 无法读取 UTF-8 文件：{exc}") from exc
    return parse_skill_content(
        content,
        source_path=path,
        origin=origin,
        max_file_chars=max_file_chars,
    )


def parse_skill_content(
    content: str,
    *,
    source_path: str | Path,
    origin: SkillOrigin = "IMPORTED",
    max_file_chars: int = 50_000,
) -> ParsedSkill:
    """Validate in-memory SKILL.md content for generated immutable candidates."""

    path = Path(source_path)
    if path.name != "SKILL.md":
        raise SkillValidationError(f"{path}: Skill 入口文件必须命名为 SKILL.md。")
    if len(content) > max_file_chars:
        raise SkillValidationError(
            f"{path}: 文件长度 {len(content)} 超过限制 {max_file_chars}。"
        )
    metadata, body = _frontmatter(content, path)

    name = metadata.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name.strip()):
        raise SkillValidationError(
            f"{path}: name 必须匹配小写 kebab-case，例如 source-verification。"
        )
    name = name.strip()

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError(f"{path}: description 必须是非空字符串。")
    description = description.strip()

    version = metadata.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SkillValidationError(f"{path}: version 必须是大于 0 的整数。")

    enabled = metadata.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SkillValidationError(f"{path}: enabled 必须是布尔值。")

    tags = _string_list(metadata.get("tags"), field="tags", path=path)
    tool_fields = [
        key
        for key in ("tools", "allowed-tools", "allowed_tools")
        if key in metadata
    ]
    if len(tool_fields) > 1:
        raise SkillValidationError(
            f"{path}: tools、allowed-tools、allowed_tools 只能使用一个。"
        )
    allowed_tools = _string_list(
        metadata.get(tool_fields[0]) if tool_fields else None,
        field=tool_fields[0] if tool_fields else "tools",
        path=path,
    )

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    skill_id = f"{name}__{content_hash[:16]}"
    record = SkillRecord(
        skill_id=skill_id,
        name=name,
        description=description,
        source_path=str(path.resolve()),
        object_path="",
        content_hash=content_hash,
        version=version,
        tags=tags,
        allowed_tools=allowed_tools,
        enabled=enabled,
        lineage=SkillLineage(origin=origin),
    )
    return ParsedSkill(record=record, raw_content=content, body=body)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def discover_skills(
    directories: tuple[str | Path, ...] | list[str | Path],
    *,
    origin: SkillOrigin = "IMPORTED",
    max_file_chars: int = 50_000,
) -> SkillDiscoveryResult:
    """Scan roots deterministically while isolating unsafe or invalid bundles."""

    skills: list[ParsedSkill] = []
    errors: list[SkillDiscoveryError] = []
    seen_paths: set[Path] = set()
    for raw_directory in directories:
        directory = Path(raw_directory)
        if not directory.exists():
            continue
        if not directory.is_dir():
            errors.append(
                SkillDiscoveryError(
                    path=str(directory),
                    error="Skill 扫描路径不是目录。",
                )
            )
            continue
        for skill_file in sorted(directory.rglob("SKILL.md")):
            resolved = skill_file.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if not _is_within(resolved, directory):
                errors.append(
                    SkillDiscoveryError(
                        path=str(skill_file),
                        error="SKILL.md 解析后位于扫描目录之外。",
                    )
                )
                continue
            try:
                skills.append(
                    parse_skill_file(
                        resolved,
                        origin=origin,
                        max_file_chars=max_file_chars,
                    )
                )
            except SkillValidationError as exc:
                errors.append(
                    SkillDiscoveryError(path=str(skill_file), error=str(exc))
                )
    return SkillDiscoveryResult(skills=tuple(skills), errors=tuple(errors))
