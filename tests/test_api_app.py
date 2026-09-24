import asyncio
import time

from fastapi.testclient import TestClient

from deepresearch.agent import ResearchResult
from deepresearch.api.app import create_app
from deepresearch.api.runs import RunManager
from deepresearch.events import ResearchEvent
from deepresearch.state import create_initial_state
from deepresearch.skill.types import SkillLineage, SkillRecord


def test_health_endpoint_reports_ready() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_query_research_run() -> None:
    async def executor(question, on_event, thread_id, skill_overrides):
        await asyncio.sleep(0)
        state = create_initial_state(question)
        state["final_report"] = "# Web 报告"
        return ResearchResult(question, "# Web 报告", state, thread_id)

    manager = RunManager(executor, thread_id_factory=lambda: "research-api-test")
    with TestClient(create_app(manager)) as client:
        created = client.post(
            "/api/runs",
            json={"question": "Web API 怎样运行？", "skill_overrides": []},
        )
        assert created.status_code == 202
        assert created.json()["thread_id"] == "research-api-test"

        payload = None
        for _ in range(50):
            response = client.get("/api/runs/research-api-test")
            payload = response.json()
            if payload["status"] == "completed":
                break
            time.sleep(0.01)

    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["final_report"] == "# Web 报告"


def test_missing_research_run_returns_404() -> None:
    with TestClient(create_app(RunManager())) as client:
        response = client.get("/api/runs/research-missing")

    assert response.status_code == 404


def test_sse_endpoint_replays_structured_events() -> None:
    async def executor(question, on_event, thread_id, skill_overrides):
        await on_event(ResearchEvent("run_started", "研究开始"))
        await on_event(ResearchEvent("run_completed", "研究完成"))
        state = create_initial_state(question)
        return ResearchResult(question, "完成", state, thread_id)

    manager = RunManager(executor, thread_id_factory=lambda: "research-sse")
    with TestClient(create_app(manager)) as client:
        client.post("/api/runs", json={"question": "测试 SSE"})
        for _ in range(50):
            if client.get("/api/runs/research-sse").json()["status"] == "completed":
                break
            time.sleep(0.01)

        response = client.get("/api/runs/research-sse/events?after=1")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 2" in response.text
    assert "event: run_completed" in response.text
    assert "研究完成" in response.text
    assert "event: run_started" not in response.text


def test_skills_endpoint_returns_public_catalog() -> None:
    skill = SkillRecord(
        skill_id="skill-1",
        name="web-research",
        description="网页研究流程",
        source_path="builtin",
        object_path="objects/skill-1.md",
        content_hash="abc",
        version=2,
        tags=("research",),
        allowed_tools=("web_search",),
        lineage=SkillLineage(origin="BUILTIN"),
    )
    with TestClient(create_app(skill_catalog=lambda: [skill])) as client:
        response = client.get("/api/skills")

    assert response.status_code == 200
    assert response.json() == [
        {
            "skill_id": "skill-1",
            "name": "web-research",
            "description": "网页研究流程",
            "version": 2,
            "origin": "BUILTIN",
            "tags": ["research"],
            "allowed_tools": ["web_search"],
            "total_selections": 0,
            "completion_rate": 0.0,
        }
    ]


def test_built_frontend_can_be_served_by_api(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "index.html").write_text(
        "<html><body>DeepResearch Workbench</body></html>",
        encoding="utf-8",
    )

    with TestClient(create_app(static_dir=tmp_path)) as client:
        response = client.get("/")
        health = client.get("/api/health")

    assert response.status_code == 200
    assert "DeepResearch Workbench" in response.text
    assert health.json() == {"status": "ok"}
