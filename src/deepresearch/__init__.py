"""Minimal deep-research agent package."""

from deepresearch.checkpointing import create_in_memory_checkpointer
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
    "create_in_memory_checkpointer",
    "run_demo",
    "run_question",
    "stream_question",
]
__version__ = "0.1.0"
