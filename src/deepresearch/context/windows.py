"""Resolve a chat model's context-window size without network access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deepresearch.context.types import WindowSource


DEFAULT_CONTEXT_WINDOW = 32_768

# Keep this table intentionally small. OpenAI-compatible services may attach
# familiar names to models with different limits, so explicit configuration
# and model-provided metadata take precedence over these known-name defaults.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    "deepseek-flash": 1_048_576,
    "deepseek-chat": 1_048_576,
    "deepseek-reasoner": 1_048_576,
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-5.6": 1_050_000,
}

_WINDOW_ATTRIBUTES = (
    "context_window",
    "max_context_window",
    "max_input_tokens",
    "model_max_tokens",
)


@dataclass(frozen=True)
class ContextWindow:
    """Resolved model identity, context capacity, and evidence source."""

    model_name: str | None
    window_tokens: int
    source: WindowSource


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _unwrap_model(model: Any) -> Any:
    """Unwrap LangChain runnable bindings while guarding against cycles."""

    current = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        bound = getattr(current, "bound", None)
        if bound is None or bound is current:
            break
        current = bound
    return current


def _identifying_params(model: Any) -> Mapping[str, Any]:
    params = getattr(model, "_identifying_params", None)
    if callable(params):
        try:
            params = params()
        except Exception:
            return {}
    return params if isinstance(params, Mapping) else {}


def resolve_model_name(model: Any) -> str | None:
    """Return the concrete model name exposed by a LangChain chat model."""

    resolved_model = _unwrap_model(model)
    for attribute in ("model_name", "model"):
        value = getattr(resolved_model, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    params = _identifying_params(resolved_model)
    for key in ("model_name", "model"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _window_from_model_metadata(model: Any) -> int | None:
    resolved_model = _unwrap_model(model)
    for attribute in _WINDOW_ATTRIBUTES:
        value = _positive_int(getattr(resolved_model, attribute, None))
        if value is not None:
            return value

    profile = getattr(resolved_model, "profile", None)
    if isinstance(profile, Mapping):
        for key in _WINDOW_ATTRIBUTES:
            value = _positive_int(profile.get(key))
            if value is not None:
                return value

    params = _identifying_params(resolved_model)
    for key in _WINDOW_ATTRIBUTES:
        value = _positive_int(params.get(key))
        if value is not None:
            return value
    return None


def _window_from_model_name(model_name: str | None) -> int | None:
    if not model_name:
        return None
    normalized = model_name.lower()
    for prefix in sorted(MODEL_CONTEXT_WINDOWS, key=len, reverse=True):
        if normalized.startswith(prefix):
            return MODEL_CONTEXT_WINDOWS[prefix]
    return None


def resolve_context_window(
    model: Any,
    explicit_window: int | None = None,
) -> ContextWindow:
    """Resolve context capacity using config, metadata, name map, then fallback."""

    model_name = resolve_model_name(model)
    if explicit_window is not None:
        configured_window = _positive_int(explicit_window)
        if configured_window is None:
            raise ValueError("显式上下文窗口必须是大于 0 的整数。")
        return ContextWindow(model_name, configured_window, "config")

    metadata_window = _window_from_model_metadata(model)
    if metadata_window is not None:
        return ContextWindow(model_name, metadata_window, "model_attribute")

    mapped_window = _window_from_model_name(model_name)
    if mapped_window is not None:
        return ContextWindow(model_name, mapped_window, "model_map")

    return ContextWindow(model_name, DEFAULT_CONTEXT_WINDOW, "fallback")
