"""Execute P1 by externalizing older large Tool results before model calls."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deepresearch.context.externalizer import ToolResultExternalizer
from deepresearch.state import ResearchState


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


def _namespace(state: ResearchState) -> str:
    try:
        configurable = get_config().get("configurable", {})
        thread_id = configurable.get("thread_id")
    except RuntimeError:
        thread_id = None
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id
    question = state.get("research_question") or "research"
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return f"run-{digest}"


class ContextExternalizationMiddleware(AgentMiddleware):
    """Replace P1-eligible ToolMessages only after their full text is persisted."""

    state_schema = ResearchState

    def __init__(self, externalizer: ToolResultExternalizer | None = None) -> None:
        self._externalizer = externalizer or ToolResultExternalizer()

    def _update(self, state: ResearchState) -> dict[str, Any] | None:
        governance = _mapping(state.get("governance"))
        context = _mapping(governance.get("context"))
        pending = context.get("pending_stages")
        if not isinstance(pending, list) or "P1" not in pending:
            return None

        batch = self._externalizer.externalize_history(
            list(state.get("messages", [])),
            namespace=_namespace(state),
        )
        previous = _mapping(context.get("externalization"))
        errors = "; ".join(batch.errors) or None
        metrics = {
            "externalized_tool_results": _non_negative_int(
                previous.get("externalized_tool_results")
            )
            + len(batch.results),
            "original_chars": _non_negative_int(previous.get("original_chars"))
            + sum(result.original_chars for result in batch.results),
            "retained_chars": _non_negative_int(previous.get("retained_chars"))
            + sum(result.retained_chars for result in batch.results),
            "estimated_tokens_saved": _non_negative_int(
                previous.get("estimated_tokens_saved")
            )
            + sum(result.estimated_tokens_saved for result in batch.results),
            "last_externalized_paths": [
                str(result.path) for result in batch.results
            ],
            "last_error": errors,
        }
        update: dict[str, Any] = {
            "governance": {"context": {"externalization": metrics}}
        }
        if batch.replacements:
            update["messages"] = batch.replacements
        return update

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Run P1 synchronously when the governance policy marks it pending."""

        return self._update(state)

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Run the same deterministic P1 pass for asynchronous graphs."""

        return self._update(state)
