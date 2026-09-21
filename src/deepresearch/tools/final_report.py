"""Tool used by the model to publish a validated Markdown research report."""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deepresearch.quality import MIN_UNIQUE_SOURCES, assess_research
from deepresearch.state import ResearchState


def prepare_final_report(
    title: str,
    report: str,
    used_source_urls: list[str],
    state: ResearchState,
) -> str:
    """Validate model-supplied report arguments and return normalized Markdown."""

    normalized_title = title.strip()
    normalized_report = report.strip()
    normalized_urls = list(
        dict.fromkeys(url.strip() for url in used_source_urls if url.strip())
    )

    if not normalized_title:
        raise ValueError("Report title cannot be empty.")
    if not normalized_report:
        raise ValueError("Report content cannot be empty.")

    readiness_gaps = assess_research(state, require_final_report=False)
    if readiness_gaps:
        raise ValueError(
            "Research is not ready for a final report: " + " ".join(readiness_gaps)
        )

    if len(normalized_urls) < MIN_UNIQUE_SOURCES:
        raise ValueError(
            f"Final report must cite at least {MIN_UNIQUE_SOURCES} unique sources."
        )

    known_urls = {source["url"] for source in state.get("sources", [])}
    unknown_urls = [url for url in normalized_urls if url not in known_urls]
    if unknown_urls:
        raise ValueError(
            "Final report contains sources that were not collected: "
            + ", ".join(unknown_urls)
        )

    missing_citations = [url for url in normalized_urls if url not in normalized_report]
    if missing_citations:
        raise ValueError(
            "Final report body must include every declared source URL: "
            + ", ".join(missing_citations)
        )

    if not normalized_report.startswith("# "):
        normalized_report = f"# {normalized_title}\n\n{normalized_report}"
    return normalized_report


@tool("write_final_report")
def write_final_report_tool(
    title: str,
    report: str,
    used_source_urls: list[str],
    runtime: ToolRuntime,
) -> Command | str:
    """Validate and store the final Markdown report after research is complete."""

    state: ResearchState = runtime.state
    try:
        final_report = prepare_final_report(
            title,
            report,
            used_source_urls,
            state,
        )
    except ValueError as exc:
        return f"Final report rejected: {exc}"

    validated_source_count = len(
        {url.strip() for url in used_source_urls if url.strip()}
    )
    return Command(
        update={
            "final_report": final_report,
            "messages": [
                ToolMessage(
                    content=(
                        "Final Markdown report saved to state with "
                        f"{validated_source_count} validated sources."
                    ),
                    tool_call_id=runtime.tool_call_id or "final-report-tool-call",
                )
            ],
        }
    )
