"""Executable capabilities exposed to the research agent."""

from deepresearch.tools.read_page import read_page_tool
from deepresearch.tools.web_search import web_search_tool

__all__ = ["read_page_tool", "web_search_tool"]
