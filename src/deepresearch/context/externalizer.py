"""Persist older large Tool results and replace them with compact references."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from deepresearch.context.tagged import (
    DEEPRESEARCH_EXTERNALIZED,
    DEEPRESEARCH_EXTERNALIZED_META,
    DEEPRESEARCH_EXTERNALIZED_PATH,
)
from deepresearch.context.tokens import estimate_context_tokens


DEFAULT_EXTERNALIZATION_DIR = Path(".deepresearch/externalized")


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(content)


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized[:80] or fallback


@dataclass(frozen=True)
class ExternalizedToolResult:
    """One safely persisted ToolMessage and its compact replacement."""

    replacement: ToolMessage
    path: Path
    original_chars: int
    retained_chars: int
    estimated_tokens_saved: int


@dataclass(frozen=True)
class ExternalizationBatch:
    """All P1 replacements and non-destructive errors from one pass."""

    results: tuple[ExternalizedToolResult, ...]
    errors: tuple[str, ...]

    @property
    def replacements(self) -> list[ToolMessage]:
        return [result.replacement for result in self.results]


class ToolResultExternalizer:
    """Externalize eligible ToolMessages while preserving recent raw results."""

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_EXTERNALIZATION_DIR,
        *,
        min_chars: int = 500,
        preview_chars: int = 400,
        preserve_recent_results: int = 2,
    ) -> None:
        if min_chars <= 0:
            raise ValueError("min_chars 必须大于 0。")
        if preview_chars <= 0:
            raise ValueError("preview_chars 必须大于 0。")
        if preserve_recent_results < 0:
            raise ValueError("preserve_recent_results 不能小于 0。")
        self._root_dir = Path(root_dir)
        self._min_chars = min_chars
        self._preview_chars = preview_chars
        self._preserve_recent_results = preserve_recent_results

    def externalize_history(
        self,
        messages: list[BaseMessage],
        *,
        namespace: str,
    ) -> ExternalizationBatch:
        """Externalize old large results, never replacing before persistence succeeds."""

        eligible = [
            message
            for message in messages
            if self._is_eligible(message)
        ]
        if self._preserve_recent_results:
            candidates = eligible[: -self._preserve_recent_results]
        else:
            candidates = eligible

        results: list[ExternalizedToolResult] = []
        errors: list[str] = []
        for message in candidates:
            try:
                results.append(self._externalize(message, namespace))
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{message.id or message.tool_call_id}: {exc}")
        return ExternalizationBatch(tuple(results), tuple(errors))

    def _is_eligible(self, message: BaseMessage) -> bool:
        return (
            isinstance(message, ToolMessage)
            and bool(message.id)
            and not message.additional_kwargs.get(DEEPRESEARCH_EXTERNALIZED)
            and len(_content_text(message.content)) >= self._min_chars
        )

    def _externalize(
        self,
        message: ToolMessage,
        namespace: str,
    ) -> ExternalizedToolResult:
        content_text = _content_text(message.content)
        content_digest = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]
        safe_namespace = _safe_component(
            namespace,
            fallback=f"namespace-{namespace_digest}",
        )
        tool_name = _safe_component(message.name or "tool", fallback="tool")
        call_id = _safe_component(message.tool_call_id, fallback="call")
        record_identity = "\n".join(
            [message.name or "", message.tool_call_id, content_digest]
        )
        record_digest = hashlib.sha256(record_identity.encode("utf-8")).hexdigest()
        directory = self._root_dir / safe_namespace
        path = directory / f"{tool_name}-{call_id}-{record_digest[:12]}.json"
        preview = content_text[: self._preview_chars]

        provisional_content = self._replacement_content(
            preview,
            path,
            original_chars=len(content_text),
            estimated_tokens_saved=0,
        )
        provisional = message.model_copy(update={"content": provisional_content})
        estimated_tokens_saved = max(
            0,
            estimate_context_tokens([message]).token_count
            - estimate_context_tokens([provisional]).token_count,
        )
        replacement_content = self._replacement_content(
            preview,
            path,
            original_chars=len(content_text),
            estimated_tokens_saved=estimated_tokens_saved,
        )
        metadata: dict[str, Any] = {
            "original_chars": len(content_text),
            "preview_chars": len(preview),
            "retained_chars": len(replacement_content),
            "estimated_tokens_saved": estimated_tokens_saved,
            "sha256": content_digest,
        }
        replacement = message.model_copy(
            update={
                "content": replacement_content,
                "additional_kwargs": {
                    **message.additional_kwargs,
                    DEEPRESEARCH_EXTERNALIZED: True,
                    DEEPRESEARCH_EXTERNALIZED_PATH: str(path),
                    DEEPRESEARCH_EXTERNALIZED_META: metadata,
                },
            }
        )
        payload = {
            "message_id": message.id,
            "tool_call_id": message.tool_call_id,
            "tool_name": message.name,
            "content": message.content,
            "content_sha256": content_digest,
            "metadata": metadata,
        }
        self._write_once(path, payload)
        return ExternalizedToolResult(
            replacement=replacement,
            path=path,
            original_chars=len(content_text),
            retained_chars=len(replacement_content),
            estimated_tokens_saved=estimated_tokens_saved,
        )

    @staticmethod
    def _replacement_content(
        preview: str,
        path: Path,
        *,
        original_chars: int,
        estimated_tokens_saved: int,
    ) -> str:
        return "\n".join(
            [
                "[P1 externalized tool result]",
                f"preview: {preview}",
                f"full_result_path: {path}",
                f"original_chars: {original_chars}",
                f"estimated_tokens_saved: {estimated_tokens_saved}",
            ]
        )

    @staticmethod
    def _write_once(path: Path, payload: dict[str, Any]) -> None:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as output_file:
                output_file.write(serialized)
                output_file.write("\n")
        except FileExistsError:
            existing = path.read_text(encoding="utf-8").rstrip("\n")
            if existing != serialized:
                raise RuntimeError(f"外化文件已存在但内容不一致：{path}") from None
