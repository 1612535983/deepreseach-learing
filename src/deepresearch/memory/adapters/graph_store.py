"""Optional association-graph contract; no concrete graph database is enabled."""

from __future__ import annotations

from typing import Protocol

from deepresearch.memory.schema import Association


class GraphStore(Protocol):
    def replace_associations(
        self,
        trace_id: str,
        associations: tuple[Association, ...],
    ) -> None: ...

    def neighbors(self, trace_id: str, *, limit: int = 20) -> list[Association]: ...
