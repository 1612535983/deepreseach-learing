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

