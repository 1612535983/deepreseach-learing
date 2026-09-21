from deepresearch.agent import ResearchResult
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
    }
    assert "任务 ID：research-cli（仅当前进程内有效）" in output
