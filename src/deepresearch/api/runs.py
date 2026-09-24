"""In-process lifecycle manager for asynchronous research runs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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
from deepresearch.checkpointing import generate_thread_id, normalize_thread_id
from deepresearch.events import ResearchEvent
from deepresearch.state import ResearchState


EventCallback = Callable[[ResearchEvent], Awaitable[None]]
RunExecutor = Callable[
    [str, EventCallback, str, tuple[str, ...]], Awaitable[ResearchResult]
]


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
        thread_id_factory: Callable[[], str] = generate_thread_id,
    ) -> None:
        self._executor = executor
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

    async def _execute(self, record: RunRecord) -> None:
        record.status = RunStatus.RUNNING
        record.updated_at = datetime.now(UTC)
        await self._notify_changed(record)

        async def on_event(event: ResearchEvent) -> None:
            await self._append_event(record, event)

        try:
            result = await self._executor(
                record.question,
                on_event,
                record.thread_id,
                record.skill_overrides,
            )
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
                        f"研究运行失败：{record.error}",
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
