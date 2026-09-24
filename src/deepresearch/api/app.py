"""FastAPI application factory for the research workbench."""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create an isolated API application for production and tests."""

    application = FastAPI(
        title="DeepResearch API",
        summary="可恢复、可审计的深度研究 Agent Web API",
        version="0.1.0",
    )

    @application.get("/api/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
