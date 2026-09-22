"""Execute P4 snapshot and summary compaction before the next model call."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deepresearch.context.snapshot import ContextSnapshotter, SnapshotResult
from deepresearch.context.summarizer import CompressionPlan, ContextSummarizer
from deepresearch.context.tokens import estimate_context_tokens
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


class ContextCompactionMiddleware(AgentMiddleware):
    """Snapshot first, then replace an old message prefix with an LLM summary."""

    state_schema = ResearchState

    def __init__(
        self,
        model: Any,
        *,
        snapshotter: ContextSnapshotter | None = None,
        summarizer: ContextSummarizer | None = None,
    ) -> None:
        self._snapshotter = snapshotter or ContextSnapshotter()
        self._summarizer = summarizer or ContextSummarizer(model)

    @staticmethod
    def _context(state: ResearchState) -> Mapping[str, Any]:
        governance = _mapping(state.get("governance"))
        return _mapping(governance.get("context"))

    def _prepare(
        self,
        state: ResearchState,
    ) -> tuple[CompressionPlan, SnapshotResult] | dict[str, Any] | None:
        context = self._context(state)
        pending = context.get("pending_stages")
        if not isinstance(pending, list) or "P4" not in pending:
            return None

        messages = list(state.get("messages", []))
        plan = self._summarizer.plan(messages)
        if plan is None:
            return None
        try:
            snapshot = self._snapshotter.create(
                state,
                messages,
                namespace=_namespace(state),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._error_update(state, f"Snapshot 失败：{exc}")
        return plan, snapshot

    def _success_update(
        self,
        state: ResearchState,
        plan: CompressionPlan,
        snapshot: SnapshotResult,
        summary: str,
    ) -> dict[str, Any]:
        context = self._context(state)
        previous = _mapping(context.get("compaction"))
        compacted = self._summarizer.compacted_messages(plan, summary, snapshot)
        messages_before = list(state.get("messages", []))
        messages_after = [compacted.summary_message, *plan.preserved]
        tokens_before = estimate_context_tokens(messages_before).token_count
        tokens_after = estimate_context_tokens(messages_after).token_count
        if tokens_after >= tokens_before:
            return self._error_update(
                state,
                "摘要没有减少 Token，已保留原消息。",
                snapshot=snapshot,
            )
        saved = tokens_before - tokens_after
        metrics = {
            "snapshot_count": _non_negative_int(previous.get("snapshot_count")) + 1,
            "summarize_count": _non_negative_int(previous.get("summarize_count"))
            + 1,
            "removed_message_count": _non_negative_int(
                previous.get("removed_message_count")
            )
            + len(plan.to_summarize),
            "estimated_tokens_saved": _non_negative_int(
                previous.get("estimated_tokens_saved")
            )
            + saved,
            "last_preserved_message_count": len(plan.preserved),
            "last_tokens_before": tokens_before,
            "last_tokens_after": tokens_after,
            "last_snapshot_path": str(snapshot.path),
            "last_summary_id": compacted.summary_id,
            "last_error": None,
        }
        return {
            "messages": compacted.patch,
            "governance": {
                "context": {
                    "summary": summary,
                    "summary_id": compacted.summary_id,
                    "compaction": metrics,
                }
            },
        }

    def _error_update(
        self,
        state: ResearchState,
        error: str,
        *,
        snapshot: SnapshotResult | None = None,
    ) -> dict[str, Any]:
        context = self._context(state)
        previous = _mapping(context.get("compaction"))
        metrics = {
            "snapshot_count": _non_negative_int(previous.get("snapshot_count"))
            + (1 if snapshot is not None else 0),
            "summarize_count": _non_negative_int(previous.get("summarize_count")),
            "removed_message_count": _non_negative_int(
                previous.get("removed_message_count")
            ),
            "estimated_tokens_saved": _non_negative_int(
                previous.get("estimated_tokens_saved")
            ),
            "last_preserved_message_count": _non_negative_int(
                previous.get("last_preserved_message_count")
            ),
            "last_tokens_before": _non_negative_int(
                previous.get("last_tokens_before")
            ),
            "last_tokens_after": _non_negative_int(previous.get("last_tokens_after")),
            "last_snapshot_path": (
                str(snapshot.path)
                if snapshot is not None
                else previous.get("last_snapshot_path")
            ),
            "last_summary_id": previous.get("last_summary_id"),
            "last_error": error,
        }
        return {"governance": {"context": {"compaction": metrics}}}

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Run the synchronous P4 flow without deleting history on any failure."""

        prepared = self._prepare(state)
        if prepared is None or isinstance(prepared, dict):
            return prepared
        plan, snapshot = prepared
        context = self._context(state)
        previous_summary = context.get("summary")
        try:
            summary = self._summarizer.summarize(
                plan,
                previous_summary=(
                    previous_summary if isinstance(previous_summary, str) else None
                ),
                snapshot_path=str(snapshot.path),
            )
        except Exception as exc:
            return self._error_update(
                state,
                f"摘要失败：{exc}",
                snapshot=snapshot,
            )
        return self._success_update(state, plan, snapshot, summary)

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Run the asynchronous P4 flow using the model's ainvoke interface."""

        prepared = self._prepare(state)
        if prepared is None or isinstance(prepared, dict):
            return prepared
        plan, snapshot = prepared
        context = self._context(state)
        previous_summary = context.get("summary")
        try:
            summary = await self._summarizer.asummarize(
                plan,
                previous_summary=(
                    previous_summary if isinstance(previous_summary, str) else None
                ),
                snapshot_path=str(snapshot.path),
            )
        except Exception as exc:
            return self._error_update(
                state,
                f"摘要失败：{exc}",
                snapshot=snapshot,
            )
        return self._success_update(state, plan, snapshot, summary)
