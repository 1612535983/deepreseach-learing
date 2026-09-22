from __future__ import annotations

import pytest

from deepresearch.context.windows import (
    DEFAULT_CONTEXT_WINDOW,
    resolve_context_window,
    resolve_model_name,
)


class NamedModel:
    model_name = "deepseek-chat"


class ModelWithWindow:
    model_name = "deepseek-chat"
    max_input_tokens = 24_000


class ModelWithProfile:
    model = "profile-model"
    profile = {"max_input_tokens": 48_000}


class ModelWithIdentifyingParams:
    @property
    def _identifying_params(self) -> dict[str, object]:
        return {
            "model_name": "params-model",
            "model_max_tokens": 96_000,
        }


class UnknownOutputLimitedModel:
    model_name = "private-compatible-model"
    max_tokens = 512


class BoundModel:
    def __init__(self, bound: object) -> None:
        self.bound = bound


def test_explicit_window_has_highest_priority() -> None:
    window = resolve_context_window(ModelWithWindow(), explicit_window=16_000)

    assert window.model_name == "deepseek-chat"
    assert window.window_tokens == 16_000
    assert window.source == "config"


@pytest.mark.parametrize("invalid_window", [0, -1, True])
def test_explicit_window_must_be_a_positive_integer(invalid_window: object) -> None:
    with pytest.raises(ValueError, match="大于 0 的整数"):
        resolve_context_window(NamedModel(), explicit_window=invalid_window)  # type: ignore[arg-type]


def test_model_attribute_has_priority_over_name_map() -> None:
    window = resolve_context_window(ModelWithWindow())

    assert window.window_tokens == 24_000
    assert window.source == "model_attribute"


def test_model_profile_can_supply_context_window() -> None:
    window = resolve_context_window(ModelWithProfile())

    assert window.model_name == "profile-model"
    assert window.window_tokens == 48_000
    assert window.source == "model_attribute"


def test_identifying_params_can_supply_name_and_window() -> None:
    model = ModelWithIdentifyingParams()

    assert resolve_model_name(model) == "params-model"
    assert resolve_context_window(model).window_tokens == 96_000


def test_known_model_name_uses_longest_prefix_mapping() -> None:
    model = NamedModel()
    model.model_name = "deepseek-v4-flash-2026-04-24"

    window = resolve_context_window(model)

    assert window.window_tokens == 1_048_576
    assert window.source == "model_map"


def test_unknown_model_uses_fallback_and_ignores_output_max_tokens() -> None:
    window = resolve_context_window(UnknownOutputLimitedModel())

    assert window.window_tokens == DEFAULT_CONTEXT_WINDOW
    assert window.window_tokens != UnknownOutputLimitedModel.max_tokens
    assert window.source == "fallback"


def test_bound_model_is_unwrapped() -> None:
    window = resolve_context_window(BoundModel(ModelWithWindow()))

    assert window.model_name == "deepseek-chat"
    assert window.window_tokens == 24_000
    assert window.source == "model_attribute"
