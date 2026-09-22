from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from deepresearch.skill.parser import parse_skill_file
from deepresearch.skill.selector import SkillSelector
from deepresearch.skill.store import SQLiteSkillStore


def _install(
    store: SQLiteSkillStore,
    root: Path,
    name: str,
    description: str,
    tags: list[str],
    tools: list[str],
):  # noqa: ANN202
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                f"tags: [{', '.join(tags)}]",
                f"tools: [{', '.join(tools)}]",
                "---",
                f"# {name}\n\nInstructions.",
            ]
        ),
        encoding="utf-8",
    )
    return store.install(parse_skill_file(path))


def _catalog(tmp_path: Path):  # noqa: ANN202
    store = SQLiteSkillStore(tmp_path / "skills.db", tmp_path / "objects")
    verify = _install(
        store,
        tmp_path,
        "source-verification",
        "核验事实声明和来源可信度",
        ["来源", "核验"],
        ["web_search", "read_page"],
    )
    report = _install(
        store,
        tmp_path,
        "evidence-report-writing",
        "根据证据撰写带引用的研究报告",
        ["报告", "引用"],
        ["write_final_report"],
    )
    generic = _install(
        store,
        tmp_path,
        "concise-answer",
        "直接简洁回答一般问题",
        ["回答"],
        [],
    )
    return store, verify, report, generic


def test_selector_ranks_relevant_chinese_skill(tmp_path: Path) -> None:
    store, verify, _, _ = _catalog(tmp_path)
    selector = SkillSelector(store, max_skills=2, min_relevance=0.01)

    selected = selector.select_for_task(
        "请核验这些来源是否支持事实声明",
        available_tools=["web_search", "read_page", "write_final_report"],
    )

    assert selected[0].record.skill_id == verify.skill_id
    assert selected[0].score > 0
    assert selected[0].reason == "bm25_metadata_match"
    store.close()


def test_selector_returns_no_irrelevant_skill(tmp_path: Path) -> None:
    store, _, _, _ = _catalog(tmp_path)
    selector = SkillSelector(store, min_relevance=0.01)

    selected = selector.select_for_task(
        "量子纠缠是什么",
        available_tools=["web_search", "read_page", "write_final_report"],
    )

    assert selected == []
    store.close()


def test_override_is_forced_and_precedes_automatic_selection(tmp_path: Path) -> None:
    store, verify, report, _ = _catalog(tmp_path)
    selector = SkillSelector(store, max_skills=2, min_relevance=0.01)

    selected = selector.select_for_task(
        "核验来源",
        overrides=[report.name],
        available_tools=["web_search", "read_page", "write_final_report"],
    )

    assert [item.record.skill_id for item in selected] == [
        report.skill_id,
        verify.skill_id,
    ]
    assert selected[0].forced is True
    assert selected[0].reason == "user_override"
    store.close()


def test_selector_filters_disabled_and_unavailable_tool_skills(tmp_path: Path) -> None:
    store, verify, report, generic = _catalog(tmp_path)
    store.set_enabled(generic.skill_id, False)
    selector = SkillSelector(store, min_relevance=0.0)

    selected = selector.select_for_task(
        "report source answer",
        available_tools=["read_page"],
    )

    ids = {item.record.skill_id for item in selected}
    assert verify.skill_id in ids
    assert report.skill_id not in ids
    assert generic.skill_id not in ids
    store.close()


def test_forced_skills_are_not_dropped_by_tool_filter_or_limit(tmp_path: Path) -> None:
    store, verify, report, _ = _catalog(tmp_path)
    selector = SkillSelector(store, max_skills=1, min_relevance=1.0)

    selected = selector.select_for_task(
        "unrelated",
        overrides=[verify.name, report.name],
        available_tools=[],
    )

    assert [item.record.name for item in selected] == [
        "source-verification",
        "evidence-report-writing",
    ]
    assert all(item.forced for item in selected)
    store.close()


def test_catalog_hash_changes_with_runtime_enable_state(tmp_path: Path) -> None:
    store, verify, _, _ = _catalog(tmp_path)
    selector = SkillSelector(store)
    before = selector.catalog_hash()

    store.set_enabled(verify.skill_id, False)

    assert selector.catalog_hash() != before
    store.close()
