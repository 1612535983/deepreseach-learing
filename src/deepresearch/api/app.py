"""FastAPI application factory for the research workbench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import json

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from deepresearch.api.runs import RunManager
from deepresearch.api.schemas import (
    CreateRunRequest,
    RunAcceptedResponse,
    RunDetailResponse,
)


def create_app(run_manager: RunManager | None = None) -> FastAPI:
    """Create an isolated API application for production and tests."""

    manager = run_manager or RunManager()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.run_manager = manager
        yield
        await manager.shutdown()

    application = FastAPI(
        title="DeepResearch API",
        summary="可恢复、可审计的深度研究 Agent Web API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/api/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/api/runs",
        response_model=RunAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["research"],
    )
    async def create_run(
        payload: CreateRunRequest,
        request: Request,
    ) -> RunAcceptedResponse:
        return await _manager(request).start(payload)

    @application.get(
        "/api/runs/{thread_id}",
        response_model=RunDetailResponse,
        tags=["research"],
    )
    async def get_run(thread_id: str, request: Request) -> RunDetailResponse:
        try:
            detail = _manager(request).detail(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(status_code=404, detail="找不到研究任务。")
        return detail

    @application.get(
        "/api/runs/{thread_id}/events",
        tags=["research"],
        response_class=StreamingResponse,
    )
    async def stream_run_events(
        thread_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        manager = _manager(request)
        try:
            record = manager.record(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="找不到研究任务。")
        cursor = after
        if last_event_id:
            try:
                cursor = max(cursor, int(last_event_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID 必须是整数。",
                ) from exc

        async def body() -> AsyncIterator[str]:
            async for event in manager.event_stream(thread_id, after=cursor):
                if await request.is_disconnected():
                    break
                if event is None:
                    yield ": keep-alive\n\n"
                    continue
                payload = json.dumps(
                    event.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                yield (
                    f"id: {event.id}\n"
                    f"event: {event.event_type}\n"
                    f"data: {payload}\n\n"
                )

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return application


def _manager(request: Request) -> RunManager:
    return request.app.state.run_manager


app = create_app()
