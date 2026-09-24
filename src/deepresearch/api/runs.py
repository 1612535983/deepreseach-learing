"""In-process lifecycle manager for asynchronous research runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from deepresearch.agent import ResearchResult, astream_question
from deepresearch.api.schemas import (
    CreateRunRequest,
    ResearchEventResponse,
    RunAcceptedResponse,
    RunDetailResponse,
    RunStatus,
    run_detail_from_state,
)
from deepresearch.agent import stream_resume_question
from deepresearch.checkpointing import (
    aget_checkpoint_state,
    generate_thread_id,
    normalize_thread_id,
    open_async_sqlite_checkpointer,
)
from deepresearch.events import ResearchEvent
from deepresearch.state import ResearchState


EventCallback = Callable[[ResearchEvent], Awaitable[None]]
RunExecutor = Callable[
    [str, EventCallback, str, tuple[str, ...]], Awaitable[ResearchResult]
]
ResumeExecutor = Callable[[EventCallback, str], Awaitable[ResearchResult]]
CheckpointLoader = Callable[[str], Awaitable[ResearchState]]


async def default_run_executor(
    question: str,
    on_event: EventCallback,
    thread_id: str,
    skill_overrides: tuple[str, ...],
) -> ResearchResult:
    """Adapt the application runner to the Web task-manager contract."""

    return await astream_question(
        question,
        on_event,
        thread_id=thread_id,
        skill_overrides=skill_overrides,
    )


async def default_resume_executor(
    on_event: EventCallback,
    thread_id: str,
) -> ResearchResult:
    """Run the existing synchronous resume path without blocking the API loop."""

    loop = asyncio.get_running_loop()

    def forward_event(event: ResearchEvent) -> None:
        future = asyncio.run_coroutine_threadsafe(on_event(event), loop)
        future.result()

    return await asyncio.to_thread(
        stream_resume_question,
        thread_id,
        forward_event,
    )


async def default_checkpoint_loader(thread_id: str) -> ResearchState:
    """Load a persisted task independently of the process-local registry."""

    async with open_async_sqlite_checkpointer() as checkpointer:
        values = await aget_checkpoint_state(checkpointer, thread_id)
    return values  # type: ignore[return-value]


@dataclass
class RunRecord:
    thread_id: str
    question: str
    skill_overrides: tuple[str, ...]
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    state: ResearchState | None = None
    error: str | None = None
    events: list[ResearchEventResponse] = field(default_factory=list)
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    changed: asyncio.Condition = field(
        default_factory=asyncio.Condition,
        repr=False,
    )

    @property
    def terminal(self) -> bool:
        return self.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }


class RunManager:
    """Own background tasks and their bounded public snapshots."""

    def __init__(
        self,
        executor: RunExecutor = default_run_executor,
        *,
        resume_executor: ResumeExecutor = default_resume_executor,
        checkpoint_loader: CheckpointLoader = default_checkpoint_loader,
        thread_id_factory: Callable[[], str] = generate_thread_id,
    ) -> None:
        self._executor = executor
        self._resume_executor = resume_executor
        self._checkpoint_loader = checkpoint_loader
        self._thread_id_factory = thread_id_factory
        self._records: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def start(self, request: CreateRunRequest) -> RunAcceptedResponse:
        thread_id = normalize_thread_id(self._thread_id_factory())
        record = RunRecord(
            thread_id=thread_id,
            question=request.question,
            skill_overrides=tuple(request.skill_overrides),
        )
        async with self._lock:
            if thread_id in self._records:
                raise RuntimeError(f"任务 ID 重复：{thread_id}")
            self._records[thread_id] = record
            record.task = asyncio.create_task(
                self._execute(record),
                name=f"deepresearch:{thread_id}",
            )
        return RunAcceptedResponse(thread_id=thread_id, status=record.status)

    async def resume(self, thread_id: str) -> RunAcceptedResponse:
        normalized = normalize_thread_id(thread_id)
        saved_state = await self._checkpoint_loader(normalized)
        question = saved_state.get("research_question")
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"任务 {normalized} 没有保存有效的研究问题。")
        async with self._lock:
            existing = self._records.get(normalized)
            if existing is not None and not existing.terminal:
                raise RuntimeError(f"任务正在运行：{normalized}")
            record = RunRecord(
                thread_id=normalized,
                question=question.strip(),
                skill_overrides=(),
                state=saved_state,
            )
            self._records[normalized] = record
            record.task = asyncio.create_task(
                self._execute_resume(record),
                name=f"deepresearch-resume:{normalized}",
            )
        return RunAcceptedResponse(thread_id=normalized, status=record.status)

    async def _execute(self, record: RunRecord) -> None:
        async def operation(on_event: EventCallback) -> ResearchResult:
            return await self._executor(
                record.question,
                on_event,
                record.thread_id,
                record.skill_overrides,
            )

        await self._run_operation(record, operation, failure_label="研究运行失败")

    async def _execute_resume(self, record: RunRecord) -> None:
        async def operation(on_event: EventCallback) -> ResearchResult:
            return await self._resume_executor(on_event, record.thread_id)

        await self._run_operation(record, operation, failure_label="研究恢复失败")

    async def _run_operation(
        self,
        record: RunRecord,
        operation: Callable[[EventCallback], Awaitable[ResearchResult]],
        *,
        failure_label: str,
    ) -> None:
        record.status = RunStatus.RUNNING
        record.updated_at = datetime.now(UTC)
        await self._notify_changed(record)

        async def on_event(event: ResearchEvent) -> None:
            await self._append_event(record, event)

        try:
            result = await operation(on_event)
        except asyncio.CancelledError:
            record.status = RunStatus.INTERRUPTED
            record.updated_at = datetime.now(UTC)
            await self._notify_changed(record)
            raise
        except Exception as exc:
            record.status = RunStatus.FAILED
            record.error = str(exc) or type(exc).__name__
            record.updated_at = datetime.now(UTC)
            if not record.events or record.events[-1].event_type != "run_failed":
                await self._append_event(
                    record,
                    ResearchEvent(
                        "run_failed",
                        f"{failure_label}：{record.error}",
                        {"error_type": type(exc).__name__},
                    ),
                )
            else:
                await self._notify_changed(record)
            return

        record.state = result.state
        record.status = RunStatus.COMPLETED
        record.updated_at = datetime.now(UTC)
        await self._notify_changed(record)

    async def _append_event(
        self,
        record: RunRecord,
        event: ResearchEvent,
    ) -> None:
        async with record.changed:
            record.events.append(
                ResearchEventResponse(
                    id=len(record.events) + 1,
                    event_type=event.event_type,
                    message=event.message,
                    data=dict(event.data),
                    node=event.node,
                    created_at=datetime.now(UTC),
                )
            )
            record.updated_at = datetime.now(UTC)
            record.changed.notify_all()

    async def _notify_changed(self, record: RunRecord) -> None:
        async with record.changed:
            record.changed.notify_all()

    def record(self, thread_id: str) -> RunRecord | None:
        return self._records.get(normalize_thread_id(thread_id))

    def detail(self, thread_id: str) -> RunDetailResponse | None:
        record = self.record(thread_id)
        if record is None:
            return None
        return run_detail_from_state(
            thread_id=record.thread_id,
            question=record.question,
            status=record.status,
            state=record.state,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
        )

    async def detail_or_checkpoint(
        self,
        thread_id: str,
    ) -> RunDetailResponse | None:
        """Return a live record or reconstruct a read-only persisted snapshot."""

        detail = self.detail(thread_id)
        if detail is not None:
            return detail
        try:
            state = await self._checkpoint_loader(thread_id)
        except ValueError:
            return None
        question = state.get("research_question")
        if not isinstance(question, str) or not question.strip():
            return None
        now = datetime.now(UTC)
        inferred_status = (
            RunStatus.COMPLETED
            if state.get("final_report")
            else RunStatus.INTERRUPTED
        )
        return run_detail_from_state(
            thread_id=normalize_thread_id(thread_id),
            question=question,
            status=inferred_status,
            state=state,
            created_at=now,
            updated_at=now,
        )

    async def event_stream(
        self,
        thread_id: str,
        *,
        after: int = 0,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[ResearchEventResponse | None]:
        """Replay stored events, then wait for new ones until the run ends.

        ``None`` is a heartbeat marker. Event IDs make browser reconnects
        resumable without duplicating the visible timeline.
        """

        record = self.record(thread_id)
        if record is None:
            raise KeyError(thread_id)
        cursor = max(0, after)
        while True:
            timed_out = False
            async with record.changed:
                if len(record.events) <= cursor and not record.terminal:
                    try:
                        await asyncio.wait_for(
                            record.changed.wait(),
                            timeout=heartbeat_seconds,
                        )
                    except TimeoutError:
                        timed_out = True
                batch = list(record.events[cursor:])
                terminal = record.terminal

            for event in batch:
                cursor = event.id
                yield event
            if terminal and cursor >= len(record.events):
                break
            if timed_out and not batch:
                yield None

    async def wait(self, thread_id: str) -> RunDetailResponse:
        record = self.record(thread_id)
        if record is None:
            raise KeyError(thread_id)
        if record.task is not None:
            try:
                await record.task
            except asyncio.CancelledError:
                pass
        detail = self.detail(thread_id)
        if detail is None:  # pragma: no cover - guarded by the record above
            raise KeyError(thread_id)
        return detail

    async def shutdown(self) -> None:
        tasks = [
            record.task
            for record in self._records.values()
            if record.task is not None and not record.task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
