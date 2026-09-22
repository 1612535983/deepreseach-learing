"""Recall relevant long-term memories before each distinct user query."""

from __future__ import annotations

import hashlib
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deepresearch.context.tagged import DEEPRESEARCH_SUMMARY
from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.context import MemoryContextRenderer
from deepresearch.memory.provider import MemoryProvider
from deepresearch.memory.types import MemoryQuery
from deepresearch.state import ResearchState


class MemoryRecallMiddleware(AgentMiddleware):
    """Retrieve once per distinct user query and store only compact references."""

    state_schema = ResearchState

    def __init__(
        self,
        provider: MemoryProvider,
        config: MemoryConfig,
    ) -> None:
        self._provider = provider
        self._config = config
        self._renderer = MemoryContextRenderer(
            provider.store(),
            token_budget=config.token_budget,
        )

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        if not self._config.enable_recall:
            return None
        query = self._latest_user_query(state)
        if not query:
            return None
        namespace = self._config.namespace.strip() or "default"
        query_hash = hashlib.sha256(
            f"{namespace}\x00{query}".encode("utf-8")
        ).hexdigest()
        memory_state = state.get("memory") or {}
        if memory_state.get("last_query_hash") == query_hash:
            return None

        try:
            results = self._provider.retriever().retrieve(
                MemoryQuery(
                    text=query,
                    namespace=namespace,
                    top_k=self._config.top_k,
                    min_strength=self._config.min_strength,
                )
            )
            rendered = self._renderer.render_results(results)
            recalled = [
                {
                    "id": result.trace.id,
                    "score": result.score,
                    "strength": result.strength,
                }
                for result in results
                if result.trace.id in rendered
            ]
            return {
                "memory": {
                    "namespace": namespace,
                    "last_query_hash": query_hash,
                    "recalled": recalled,
                    "recall_count": int(memory_state.get("recall_count", 0)) + 1,
                    "injected_tokens": self._renderer.token_count(rendered),
                    "last_error": None,
                }
            }
        except Exception as exc:
            return {
                "memory": {
                    "namespace": namespace,
                    "last_query_hash": query_hash,
                    "recalled": [],
                    "injected_tokens": 0,
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            }

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    @staticmethod
    def _latest_user_query(state: ResearchState) -> str:
        for message in reversed(state.get("messages", [])):
            if not isinstance(message, HumanMessage):
                continue
            if message.additional_kwargs.get(DEEPRESEARCH_SUMMARY):
                continue
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        return item["text"].strip()
            return str(content).strip()
        return ""
