import asyncio

from deepresearch.agent import ResearchResult
from deepresearch.api.runs import RunManager
from deepresearch.api.schemas import CreateRunRequest, RunStatus
from deepresearch.events import ResearchEvent
from deepresearch.state import create_initial_state


def test_run_manager_tracks_successful_lifecycle() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def executor(question, on_event, thread_id, skill_overrides):
            assert skill_overrides == ("web-research",)
            await on_event(
                ResearchEvent("run_started", f"研究开始：{question}")
            )
            started.set()
            await release.wait()
            state = create_initial_state(question)
            state["final_report"] = "# 已完成"
            await on_event(ResearchEvent("run_completed", "研究任务已完成"))
            return ResearchResult(question, "# 已完成", state, thread_id)

        manager = RunManager(executor, thread_id_factory=lambda: "research-web-test")
        accepted = await manager.start(
            CreateRunRequest(
                question="研究 Web API",
                skill_overrides=["web-research"],
            )
        )
        await started.wait()

        running = manager.detail(accepted.thread_id)
        assert running is not None
        assert running.status == RunStatus.RUNNING

        release.set()
        completed = await manager.wait(accepted.thread_id)

        assert completed.status == RunStatus.COMPLETED
        assert completed.final_report == "# 已完成"
        assert [event.event_type for event in manager.record(accepted.thread_id).events] == [
            "run_started",
            "run_completed",
        ]
        await manager.shutdown()

    asyncio.run(scenario())


def test_run_manager_turns_executor_error_into_failed_task() -> None:
    async def scenario() -> None:
        async def executor(question, on_event, thread_id, skill_overrides):
            raise RuntimeError("模型服务不可用")

        manager = RunManager(executor, thread_id_factory=lambda: "research-failed")
        accepted = await manager.start(CreateRunRequest(question="测试失败"))
        failed = await manager.wait(accepted.thread_id)

        assert failed.status == RunStatus.FAILED
        assert failed.error == "模型服务不可用"
        assert manager.record(accepted.thread_id).events[-1].event_type == "run_failed"

    asyncio.run(scenario())
