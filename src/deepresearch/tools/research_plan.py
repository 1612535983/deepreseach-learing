"""State-mutating tools used by the model to create and advance a research plan."""

from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deepresearch.state import PlanStatus, PlanStep, ResearchPlan, ResearchState


def _tool_message(runtime: ToolRuntime, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id or "planning-tool-call",
    )


@tool("write_research_plan")
def write_research_plan_tool(
    goal: str,
    steps: list[str],
    runtime: ToolRuntime,
) -> Command | str:
    """Create an ordered research plan before searching or reading pages."""

    normalized_goal = goal.strip()
    normalized_steps = [step.strip() for step in steps if step.strip()]
    if not normalized_goal:
        return "Research plan goal cannot be empty."
    if not 2 <= len(normalized_steps) <= 5:
        return "Research plan must contain 2 to 5 non-empty steps."

    plan_steps: list[PlanStep] = [
        {
            "step_id": f"step-{index}",
            "title": title,
            "status": "in_progress" if index == 1 else "pending",
        }
        for index, title in enumerate(normalized_steps, start=1)
    ]
    plan: ResearchPlan = {"goal": normalized_goal, "steps": plan_steps}
    return Command(
        update={
            "plan": plan,
            "current_step_id": "step-1",
            "messages": [
                _tool_message(
                    runtime,
                    f"Research plan created with {len(plan_steps)} steps; step-1 is in progress.",
                )
            ],
        }
    )


@tool("update_plan_step")
def update_plan_step_tool(
    step_id: str,
    status: Literal["pending", "in_progress", "completed", "failed"],
    runtime: ToolRuntime,
) -> Command | str:
    """Update one research step and select the next pending step when needed."""

    state: ResearchState = runtime.state
    existing_plan = state.get("plan")
    if not existing_plan:
        return "No research plan exists. Call write_research_plan first."

    matched = False
    updated_steps: list[PlanStep] = []
    for step in existing_plan["steps"]:
        next_status: PlanStatus = step["status"]
        if step["step_id"] == step_id:
            matched = True
            next_status = status
        elif status == "in_progress" and step["status"] == "in_progress":
            next_status = "pending"
        updated_steps.append({**step, "status": next_status})
    if not matched:
        return f"Unknown plan step: {step_id}"

    current_step_id: str | None = None
    if status == "in_progress":
        current_step_id = step_id
    else:
        for index, step in enumerate(updated_steps):
            if step["status"] == "in_progress":
                current_step_id = step["step_id"]
                break
            if step["status"] == "pending":
                updated_steps[index] = {**step, "status": "in_progress"}
                current_step_id = step["step_id"]
                break

    updated_plan: ResearchPlan = {
        "goal": existing_plan["goal"],
        "steps": updated_steps,
    }
    progress_message = (
        f"Plan step {step_id} updated to {status}; current step is {current_step_id}."
        if current_step_id
        else f"Plan step {step_id} updated to {status}; no pending steps remain."
    )
    return Command(
        update={
            "plan": updated_plan,
            "current_step_id": current_step_id,
            "messages": [_tool_message(runtime, progress_message)],
        }
    )
