"""Structured state shared by the research agent graph.

The message history remains LangChain's responsibility. The additional fields
form a small, typed research ledger that later middleware can populate from
tool results.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast
from uuid import uuid4

from langchain.agents import AgentState
from langchain_core.messages import HumanMessage

from deepresearch.context.types import GovernanceState, TaggedContextState
from deepresearch.memory.types import MemoryRuntimeState
from deepresearch.skill.types import SkillRuntimeState


class SearchRecord(TypedDict):
    """One search attempt, whether it succeeded or failed."""

    query: str
    success: bool
    result_count: int
    error: str | None
    step_id: NotRequired[str]


class Source(TypedDict):
    """A normalized web source discovered by a search."""

    title: str
    url: str
    snippet: str
    query: str


class PageRecord(TypedDict):
    """One attempt to fetch and extract a web page."""

    requested_url: str
    final_url: str | None
    success: bool
    content_chars: int
    truncated: bool
    error: str | None
    step_id: NotRequired[str]


class Observation(TypedDict):
    """A piece of evidence that may support the final answer."""

    content: str
    source_url: str
    query: str
    evidence_type: NotRequired[str]
    step_id: NotRequired[str]


PlanStatus = Literal["pending", "in_progress", "completed", "failed"]


class PlanStep(TypedDict):
    """One explicit unit of work in the research plan."""

    step_id: str
    title: str
    status: PlanStatus


class ResearchPlan(TypedDict):
    """The structured goal and ordered steps for one research run."""

    goal: str
    steps: list[PlanStep]


def append_search_records(
    current: list[SearchRecord] | None,
    incoming: list[SearchRecord] | None,
) -> list[SearchRecord]:
    """Append new search attempts without discarding earlier attempts."""

    return list(current or []) + list(incoming or [])


def merge_sources(
    current: list[Source] | None,
    incoming: list[Source] | None,
) -> list[Source]:
    """Merge sources by URL; newer metadata replaces older metadata."""

    merged = list(current or [])
    positions = {source["url"]: index for index, source in enumerate(merged)}
    for source in incoming or []:
        url = source["url"].strip()
        if not url:
            continue
        if url in positions:
            existing = merged[positions[url]]
            merged[positions[url]] = {
                "title": source["title"] or existing["title"],
                "url": url,
                "snippet": source["snippet"] or existing["snippet"],
                "query": source["query"] or existing["query"],
            }
        else:
            positions[url] = len(merged)
            merged.append(source)
    return merged


def append_observations(
    current: list[Observation] | None,
    incoming: list[Observation] | None,
) -> list[Observation]:
    """Append evidence in discovery order."""

    return list(current or []) + list(incoming or [])


def append_page_records(
    current: list[PageRecord] | None,
    incoming: list[PageRecord] | None,
) -> list[PageRecord]:
    """Append page reads without discarding earlier attempts."""

    return list(current or []) + list(incoming or [])


def merge_final_report(current: str | None, incoming: str | None) -> str | None:
    """Keep the existing report unless a new non-null report is supplied."""

    return incoming if incoming is not None else current


def merge_plan(
    current: ResearchPlan | None,
    incoming: ResearchPlan | None,
) -> ResearchPlan | None:
    """Replace the plan with an explicit update, preserving it on null patches."""

    return incoming if incoming is not None else current


def merge_governance(
    current: GovernanceState | None,
    incoming: GovernanceState | None,
) -> GovernanceState:
    """Deep-merge partial governance patches without mutating either input."""

    def merge_dict(
        target: dict[str, Any],
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        for key, incoming_value in patch.items():
            current_value = target.get(key)
            if isinstance(current_value, dict) and isinstance(incoming_value, dict):
                target[key] = merge_dict(current_value, incoming_value)
            else:
                target[key] = deepcopy(incoming_value)
        return target

    merged = merge_dict(deepcopy(dict(current or {})), dict(incoming or {}))
    return cast(GovernanceState, merged)


def merge_tagged_context(
    current: TaggedContextState | None,
    incoming: TaggedContextState | None,
) -> TaggedContextState | None:
    """Keep the latest detached model-context audit snapshot."""

    if incoming is None:
        return deepcopy(current)
    return deepcopy(incoming)


def merge_memory_runtime(
    current: MemoryRuntimeState | None,
    incoming: MemoryRuntimeState | None,
) -> MemoryRuntimeState:
    """Merge a partial runtime patch while replacing the current recall list."""

    merged = deepcopy(dict(current or {}))
    for key, value in dict(incoming or {}).items():
        merged[key] = deepcopy(value)
    return cast(MemoryRuntimeState, merged)


def merge_skill_runtime(
    current: SkillRuntimeState | None,
    incoming: SkillRuntimeState | None,
) -> SkillRuntimeState:
    """Merge partial skill patches while replacing selection lists explicitly."""

    merged = deepcopy(dict(current or {}))
    for key, value in dict(incoming or {}).items():
        merged[key] = deepcopy(value)
    return cast(SkillRuntimeState, merged)


def create_initial_memory_state(namespace: str = "default") -> MemoryRuntimeState:
    """Create the checkpoint-safe runtime view for long-term memory."""

    normalized_namespace = namespace.strip() or "default"
    return {
        "namespace": normalized_namespace,
        "last_query_hash": None,
        "recalled": [],
        "recall_count": 0,
        "injected_tokens": 0,
        "render_signature": None,
        "pending_write_count": 0,
        "last_error": None,
        "last_processed_run": None,
    }


def create_initial_skill_state() -> SkillRuntimeState:
    """Create the complete checkpoint-safe skill namespace for a new run."""

    return {
        "run_id": uuid4().hex,
        "query_hash": None,
        "catalog_hash": None,
        "selected": [],
        "dropped": [],
        "selection_count": 0,
        "injection_count": 0,
        "injected_tokens": 0,
        "render_signature": None,
        "aligned_tool_calls": 0,
        "completed_recorded": False,
        "last_error": None,
    }


def create_initial_governance_state() -> GovernanceState:
    """Create the complete, serializable governance namespace for a new run."""

    return {
        "context": {
            "model_name": None,
            "budget": {
                "current_tokens": 0,
                "window_tokens": 0,
                "utilization_ratio": 0.0,
                "token_count_method": "unknown",
                "window_source": "unknown",
            },
            "cumulative_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "model_call_count": 0,
            "pending_stages": [],
            "hard_limit_reached": False,
            "seen_message_usage": {},
            "externalization": {
                "externalized_tool_results": 0,
                "original_chars": 0,
                "retained_chars": 0,
                "estimated_tokens_saved": 0,
                "last_externalized_paths": [],
                "last_error": None,
            },
            "summary": None,
            "summary_id": None,
            "compaction": {
                "snapshot_count": 0,
                "summarize_count": 0,
                "removed_message_count": 0,
                "estimated_tokens_saved": 0,
                "last_preserved_message_count": 0,
                "last_tokens_before": 0,
                "last_tokens_after": 0,
                "last_snapshot_path": None,
                "last_summary_id": None,
                "last_error": None,
            },
            "finalization": {
                "active": False,
                "redirect_count": 0,
                "blocked_tool_call_count": 0,
                "terminal_tool_call_count": 0,
                "forced_stop_count": 0,
                "last_blocked_tool_names": [],
                "last_utilization_ratio": 0.0,
                "last_reason": None,
                "last_reminder_id": None,
            },
        }
    }


class ResearchState(AgentState):
    """The shared blackboard for one research graph execution."""

    research_question: NotRequired[str]
    search_records: Annotated[list[SearchRecord], append_search_records]
    page_records: Annotated[list[PageRecord], append_page_records]
    sources: Annotated[list[Source], merge_sources]
    observations: Annotated[list[Observation], append_observations]
    plan: Annotated[ResearchPlan | None, merge_plan]
    current_step_id: NotRequired[str | None]
    reflection_attempts: NotRequired[int]
    research_gaps: NotRequired[list[str]]
    final_report: Annotated[str | None, merge_final_report]
    governance: Annotated[GovernanceState, merge_governance]
    tagged_context: Annotated[TaggedContextState | None, merge_tagged_context]
    memory: Annotated[MemoryRuntimeState, merge_memory_runtime]
    skills: Annotated[SkillRuntimeState, merge_skill_runtime]


def create_initial_state(
    question: str,
    *,
    memory_namespace: str = "default",
) -> ResearchState:
    """Create the complete state passed into the first graph node."""

    return {
        "messages": [HumanMessage(content=question)],
        "research_question": question,
        "search_records": [],
        "page_records": [],
        "sources": [],
        "observations": [],
        "plan": None,
        "current_step_id": None,
        "reflection_attempts": 0,
        "research_gaps": [],
        "final_report": None,
        "governance": create_initial_governance_state(),
        "tagged_context": None,
        "memory": create_initial_memory_state(memory_namespace),
        "skills": create_initial_skill_state(),
    }
