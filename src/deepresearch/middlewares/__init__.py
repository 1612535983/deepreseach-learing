"""Lifecycle hooks that add research behavior around the core agent loop."""

from deepresearch.middlewares.evidence import EvidenceMiddleware

__all__ = ["EvidenceMiddleware"]

