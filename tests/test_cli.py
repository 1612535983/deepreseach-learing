from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import ResearchResult, run_with_model
from deepresearch.checkpointing import open_sqlite_checkpointer
from deepresearch.cli import main
from deepresearch.events import ResearchEvent
from deepresearch.state import create_initial_state


def test_run_can_show_research_trace(monkeypatch, capsys) -> None:  # noqa: ANN001
    state = create_initial_state("测试问题")
    state["search_records"] = [
        {
            "query": "test query",
            "success": True,
            "result_count": 1,
            "error": None,
        }
    ]
    result = ResearchResult(question="测试问题", answer="最终答案", state=state)
    monkeypatch.setattr(
        "deepresearch.cli.run_question",
        lambda question, **kwargs: result,
    )

    exit_code = main(["run", "测试问题", "--show-trace"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "最终答案" in output
    assert "搜索次数：1" in output
    assert "test query" in output


def test_run_can_save_final_report(monkeypatch, capsys, tmp_path) -> None:  # noqa: ANN001
    state = create_initial_state("测试问题")
    state["final_report"] = "# 最终研究报告\n\n报告正文"
    result = ResearchResult(
        question="测试问题",
        answer=state["final_report"],
        state=state,
    )
    monkeypatch.setattr(
        "deepresearch.cli.run_question",
        lambda question, **kwargs: result,
    )
    report_path = tmp_path / "reports" / "result.md"

    exit_code = main(["run", "测试问题", "--output", str(report_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert report_path.read_text(encoding="utf-8") == "# 最终研究报告\n\n报告正文\n"
    assert f"报告已保存：{report_path}" in output


def test_run_can_display_stream_events(monkeypatch, capsys) -> None:  # noqa: ANN001
    state = create_initial_state("测试问题")
    result = ResearchResult(question="测试问题", answer="最终答案", state=state)

    def fake_stream(question, on_event, **kwargs):  # noqa: ANN001, ANN003, ANN202
        on_event(ResearchEvent("run_started", f"研究开始：{question}"))
        on_event(
            ResearchEvent(
                "search_completed",
                "搜索完成：test query，返回 2 个结果",
            )
        )
        on_event(ResearchEvent("run_completed", "研究任务已完成"))
        return result

    monkeypatch.setattr("deepresearch.cli.stream_question", fake_stream)

    exit_code = main(["run", "测试问题", "--stream"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[开始] 研究开始：测试问题" in output
    assert "[搜索] 搜索完成：test query，返回 2 个结果" in output
    assert "[完成] 研究任务已完成" in output
    assert "最终答案" in output


def test_run_passes_thread_id_and_displays_it(monkeypatch, capsys) -> None:  # noqa: ANN001
    state = create_initial_state("测试问题")
    result = ResearchResult(
        question="测试问题",
        answer="最终答案",
        state=state,
        thread_id="research-cli",
    )
    received: dict[str, str | None] = {}

    def fake_run(question, **kwargs):  # noqa: ANN001, ANN003, ANN202
        received["question"] = question
        received["thread_id"] = kwargs.get("thread_id")
        received["skill_overrides"] = kwargs.get("skill_overrides")  # type: ignore[assignment]
        return result

    monkeypatch.setattr("deepresearch.cli.run_question", fake_run)

    exit_code = main(
        ["run", "测试问题", "--thread-id", "research-cli"]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert received == {
        "question": "测试问题",
        "thread_id": "research-cli",
        "skill_overrides": (),
    }
    assert "任务 ID：research-cli（已持久化到 SQLite）" in output


def test_resume_passes_thread_id_and_can_stream(monkeypatch, capsys) -> None:  # noqa: ANN001
    state = create_initial_state("恢复问题")
    result = ResearchResult(
        question="恢复问题",
        answer="恢复答案",
        state=state,
        thread_id="resume-cli",
    )
    received: dict[str, str] = {}

    def fake_resume(thread_id, on_event, **kwargs):  # noqa: ANN001, ANN003, ANN202
        received["thread_id"] = thread_id
        on_event(ResearchEvent("run_started", "恢复研究：恢复问题"))
        return result

    monkeypatch.setattr("deepresearch.cli.stream_resume_question", fake_resume)

    exit_code = main(["resume", "resume-cli", "--stream"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert received == {"thread_id": "resume-cli"}
    assert "[开始] 恢复研究：恢复问题" in output
    assert "恢复答案" in output


def test_inspect_reads_sqlite_without_model(monkeypatch, capsys, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    with open_sqlite_checkpointer() as checkpointer:
        run_with_model(
            "检查问题",
            FakeMessagesListChatModel(responses=[AIMessage(content="检查答案")]),
            checkpointer=checkpointer,
            thread_id="inspect-cli",
        )

    exit_code = main(["inspect", "inspect-cli"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "任务 ID：inspect-cli" in output
    assert "研究问题：检查问题" in output
    assert "Graph 步骤：" in output
    assert "最终报告：未生成" in output
    assert "上下文治理：" in output
    assert "上下文窗口：32,768 Token（来源：fallback）" in output
    assert "当前上下文：" in output
    assert "上下文占用：" in output
    assert "累计 Token：0（输入 0 / 输出 0）" in output
    assert "模型调用次数：1" in output
    assert "待处理阶段：无" in output
    assert "P1 外化：0 个 Tool 结果；预计节省 0 Token" in output
    assert "P4 压缩：0 次摘要；0 个快照；累计移除 0 条消息" in output
    assert "P5/研究预算收尾：未触发；重定向 0 次；拦截 0 个 Tool Call" in output
    assert "长期记忆：" in output
    assert "当前召回数量：0" in output
    assert "Skills：" in output
    assert "选中数量：0" in output
    assert "长期记忆库：未启用" in output


def test_inspect_reports_missing_thread(monkeypatch, capsys, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)

    exit_code = main(["inspect", "missing-thread"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "错误：找不到任务：missing-thread" in output


def test_run_passes_explicit_skill_overrides(monkeypatch, capsys) -> None:  # noqa: ANN001
    result = ResearchResult(
        question="测试问题",
        answer="答案",
        state=create_initial_state("测试问题"),
    )
    received = {}

    def fake_run(question, **kwargs):  # noqa: ANN001, ANN003, ANN202
        received.update(kwargs)
        return result

    monkeypatch.setattr("deepresearch.cli.run_question", fake_run)

    assert main(["run", "测试问题", "--skill", "verify", "--skill", "report"]) == 0
    assert received["skill_overrides"] == ("verify", "report")
    assert "答案" in capsys.readouterr().out


def test_skills_cli_validates_lists_shows_and_disables(
    monkeypatch, capsys, tmp_path
) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    skill_dir = tmp_path / "skills" / "verify"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: verify
description: 核验来源
tags: [verification]
tools: [read_page]
---
优先查看一手来源。
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPRESEARCH_SKILL_INCLUDE_BUILTIN", "false")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_DIRS", str(tmp_path / "skills"))
    monkeypatch.setenv("DEEPRESEARCH_SKILL_DB_PATH", str(tmp_path / "skills.db"))
    monkeypatch.setenv(
        "DEEPRESEARCH_SKILL_STORAGE_PATH", str(tmp_path / "objects")
    )

    assert main(["skills", "validate"]) == 0
    assert "校验完成：1 个有效，0 个无效" in capsys.readouterr().out
    assert main(["skills", "list"]) == 0
    list_output = capsys.readouterr().out
    assert "verify [verify__" in list_output
    first_id = list_output.split("[", 1)[1].split("]", 1)[0]
    assert main(["skills", "show", "verify"]) == 0
    assert "优先查看一手来源" in capsys.readouterr().out
    assert main(["skills", "disable", "verify"]) == 0
    assert "已停用：verify" in capsys.readouterr().out
    assert main(["skills", "list"]) == 0
    assert "disabled" in capsys.readouterr().out
    assert main(["skills", "history", "verify"]) == 0
    assert "active" in capsys.readouterr().out

    (skill_dir / "SKILL.md").write_text(
        """---
name: verify
description: 核验来源第二版
version: 2
---
使用更新后的核验流程。
""",
        encoding="utf-8",
    )
    assert main(["skills", "list"]) == 0
    assert "v2 enabled" in capsys.readouterr().out
    assert main(["skills", "history", "verify"]) == 0
    assert capsys.readouterr().out.count("verify [verify__") == 2
    assert main(["skills", "rollback", first_id]) == 0
    assert f"已切换激活版本：{first_id}" in capsys.readouterr().out
    assert main(["skills", "history", "verify"]) == 0
    history_output = capsys.readouterr().out
    assert f"[{first_id}] v1 disabled" in history_output
    assert history_output.splitlines()[0].endswith("* active")
