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


def test_event_stream_replays_events_and_finishes_with_run() -> None:
    async def scenario() -> None:
        async def executor(question, on_event, thread_id, skill_overrides):
            await on_event(ResearchEvent("run_started", "开始"))
            await on_event(ResearchEvent("report_created", "报告完成"))
            state = create_initial_state(question)
            state["final_report"] = "# 完成"
            return ResearchResult(question, "# 完成", state, thread_id)

        manager = RunManager(executor, thread_id_factory=lambda: "research-events")
        accepted = await manager.start(CreateRunRequest(question="事件流"))
        await manager.wait(accepted.thread_id)

        events = [
            event
            async for event in manager.event_stream(
                accepted.thread_id,
                after=1,
                heartbeat_seconds=0.01,
            )
        ]

        assert len(events) == 1
        assert events[0] is not None
        assert events[0].id == 2
        assert events[0].event_type == "report_created"

    asyncio.run(scenario())


def test_resume_uses_persisted_question_and_resume_executor() -> None:
    async def scenario() -> None:
        saved = create_initial_state("恢复这个问题")

        async def load_checkpoint(thread_id):
            assert thread_id == "research-resume"
            return saved

        async def resume_executor(on_event, thread_id):
            await on_event(ResearchEvent("run_started", "恢复研究"))
            restored = create_initial_state("恢复这个问题")
            restored["final_report"] = "# 已恢复"
            return ResearchResult("恢复这个问题", "# 已恢复", restored, thread_id)

        manager = RunManager(
            resume_executor=resume_executor,
            checkpoint_loader=load_checkpoint,
        )
        accepted = await manager.resume("research-resume")
        completed = await manager.wait(accepted.thread_id)

        assert completed.status == RunStatus.COMPLETED
        assert completed.question == "恢复这个问题"
        assert completed.final_report == "# 已恢复"

    asyncio.run(scenario())


def test_detail_falls_back_to_checkpoint_after_process_restart() -> None:
    async def scenario() -> None:
        saved = create_initial_state("持久化问题")
        saved["final_report"] = "# 持久化报告"

        async def load_checkpoint(thread_id):
            return saved

        manager = RunManager(checkpoint_loader=load_checkpoint)
        detail = await manager.detail_or_checkpoint("research-persisted")

        assert detail is not None
        assert detail.status == RunStatus.COMPLETED
        assert detail.final_report == "# 持久化报告"

    asyncio.run(scenario())
