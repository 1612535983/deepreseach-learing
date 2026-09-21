from deepresearch.agent import ResearchResult
from deepresearch.cli import main
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
    monkeypatch.setattr("deepresearch.cli.run_question", lambda question: result)

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
    monkeypatch.setattr("deepresearch.cli.run_question", lambda question: result)
    report_path = tmp_path / "reports" / "result.md"

    exit_code = main(["run", "测试问题", "--output", str(report_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert report_path.read_text(encoding="utf-8") == "# 最终研究报告\n\n报告正文\n"
    assert f"报告已保存：{report_path}" in output
