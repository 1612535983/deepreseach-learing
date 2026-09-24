from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deepresearch.api.schemas import (
    CreateRunRequest,
    RunStatus,
    run_detail_from_state,
)
from deepresearch.state import create_initial_state


def test_create_run_request_normalizes_question_and_skills() -> None:
    request = CreateRunRequest(
        question="  研究 Agent 可靠性  ",
        skill_overrides=[" web-research ", "", "web-research"],
    )

    assert request.question == "研究 Agent 可靠性"
    assert request.skill_overrides == ["web-research"]


def test_create_run_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        CreateRunRequest(question="   ")


def test_run_detail_projects_only_public_bounded_state() -> None:
    state = create_initial_state("JEV 如何工作？")
    state["plan"] = {
        "goal": "解释 JEV",
        "steps": [{"step_id": "s1", "title": "收集资料", "status": "completed"}],
    }
    state["sources"] = [
        {"title": "官方文档", "url": "https://example.com", "snippet": "摘要", "query": "JEV"}
    ]
    state["observations"] = [
        {"content": "不应出现在公开响应里的长证据", "source_url": "https://example.com", "query": "JEV"}
    ]
    state["final_report"] = "# JEV 报告"
    state["skills"]["selected"] = [
        {
            "skill_id": "skill-1",
            "name": "source-verification",
            "score": 0.9,
            "reason": "相关",
            "forced": False,
        }
    ]
    state["evaluation"]["report"].update(
        {"status": "completed", "composite_score": 0.82}
    )
    now = datetime.now(UTC)

    detail = run_detail_from_state(
        thread_id="research-test",
        question="JEV 如何工作？",
        status=RunStatus.COMPLETED,
        state=state,
        created_at=now,
        updated_at=now,
    )
    payload = detail.model_dump(mode="json")

    assert payload["plan"]["steps"][0]["status"] == "completed"
    assert payload["progress"]["source_count"] == 1
    assert payload["progress"]["evidence_count"] == 1
    assert payload["selected_skills"][0]["name"] == "source-verification"
    assert payload["evaluation"]["composite_score"] == 0.82
    assert payload["final_report"] == "# JEV 报告"
    assert "messages" not in payload
    assert "observations" not in payload
    assert "不应出现在公开响应" not in str(payload)
