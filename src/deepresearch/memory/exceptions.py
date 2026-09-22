"""Domain-specific errors raised by long-term memory components."""


class MemoryErrorBase(RuntimeError):
    """Base class for memory errors callers may handle without hiding bugs."""


class MemoryNotFoundError(MemoryErrorBase):
    def __init__(self, trace_id: str) -> None:
        super().__init__(f"找不到记忆：{trace_id}")
        self.trace_id = trace_id


class MemoryConflictError(MemoryErrorBase):
    def __init__(self, trace_id: str) -> None:
        super().__init__(f"记忆已存在：{trace_id}")
        self.trace_id = trace_id
