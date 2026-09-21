"""Minimal deep-research agent package."""

from deepresearch.agent import (
    ResearchResult,
    astream_question,
    run_demo,
    run_question,
    stream_question,
)
from deepresearch.events import ResearchEvent

__all__ = [
    "ResearchEvent",
    "ResearchResult",
    "astream_question",
    "run_demo",
    "run_question",
    "stream_question",
]
__version__ = "0.1.0"
