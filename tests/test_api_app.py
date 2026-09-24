import asyncio
import time

from fastapi.testclient import TestClient

from deepresearch.agent import ResearchResult
from deepresearch.api.app import create_app
from deepresearch.api.runs import RunManager
from deepresearch.state import create_initial_state


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
