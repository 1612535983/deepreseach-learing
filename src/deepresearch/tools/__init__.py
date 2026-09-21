"""Executable capabilities exposed to the research agent."""

from deepresearch.tools.final_report import write_final_report_tool
from deepresearch.tools.read_page import read_page_tool
from deepresearch.tools.research_plan import (
    update_plan_step_tool,
    write_research_plan_tool,
)
from deepresearch.tools.web_search import web_search_tool

__all__ = [
    "read_page_tool",
    "update_plan_step_tool",
    "web_search_tool",
    "write_final_report_tool",
    "write_research_plan_tool",
]
