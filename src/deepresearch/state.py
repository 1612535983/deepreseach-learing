"""Structured state shared by the research agent graph.

The message history remains LangChain's responsibility. The additional fields
form a small, typed research ledger that later middleware can populate from
tool results.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import HumanMessage


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


def create_initial_state(question: str) -> ResearchState:
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
    }
