"""FastAPI application factory for the research workbench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status

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

    return application


def _manager(request: Request) -> RunManager:
    return request.app.state.run_manager


app = create_app()
