from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import run_demo, run_with_model
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import SkillManager


def test_offline_demo_runs_complete_agent_graph() -> None:
    result = run_demo("测试问题")

    assert result.question == "测试问题"
    assert "最小链路已跑通" in result.answer
    assert result.state["research_question"] == "测试问题"
    assert result.state["search_records"] == []


def test_agent_returns_model_answer() -> None:
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="这是模型答案")]
    )

    result = run_with_model("什么是 ReAct？", model)

    assert result.question == "什么是 ReAct？"
    assert result.answer == "这是模型答案"
    assert result.state["sources"] == []
    assert result.state["messages"][-1].content == "这是模型答案"
    assert result.state["tagged_context"] is not None
    assert "<goal>什么是 ReAct？</goal>" in result.state["tagged_context"]["rendered"]


def test_agent_selects_and_injects_versioned_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "source-verification"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: source-verification
description: 核验来源
tags: [verification]
---
# 来源核验

优先使用一手来源，并交叉核验关键结论。
""",
        encoding="utf-8",
    )
    config = SkillConfig(
        use="default",
        db_path=tmp_path / "skills.db",
        storage_path=tmp_path / "objects",
        skill_dirs=(tmp_path / "skills",),
        include_builtin=False,
        min_relevance=0.01,
    )
    manager = SkillManager(config)
    manager.load_startup()
    model = FakeMessagesListChatModel(responses=[AIMessage(content="已核验")])

    result = run_with_model(
        "请核验这个来源",
        model,
        skill_manager=manager,
        skill_config=config,
    )

    selected = result.state["skills"]["selected"]
    assert [item["name"] for item in selected] == ["source-verification"]
    assert result.state["skills"]["selection_count"] == 1
    assert result.state["skills"]["injection_count"] == 1
    assert result.state["skills"]["completed_recorded"] is True
    rendered = result.state["tagged_context"]["rendered"]
    assert '<skill name="source-verification"' in rendered
    assert "优先使用一手来源" in rendered
    assert not any(message.type == "system" for message in result.state["messages"])
    metrics = manager.store.get_metrics(selected[0]["skill_id"])
    assert metrics is not None
    assert metrics.selections == 1
    assert metrics.injections == 1
    assert metrics.completed_runs == 1
    manager.close()
