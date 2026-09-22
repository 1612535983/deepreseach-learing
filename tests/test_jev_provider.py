from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.exceptions import (
    EvaluationProviderError,
    EvaluationResponseError,
)
from deepresearch.evaluation.jev import JevDecisionProvider
from deepresearch.evaluation.types import DecisionQuestion


QUESTIONS = {
    "supported": DecisionQuestion(
        "noul",
        "Does `evidence` support `report`?",
    ),
    "quality": DecisionQuestion(
        "score",
        "Rate source quality.",
        ("weak", "mixed", "authoritative"),
    ),
}


def _config(**updates) -> EvaluationConfig:  # noqa: ANN003
    values = {
        "use": "jev",
        "api_key": "test-secret",
        "timeout_seconds": 1.0,
        "max_retries": 1,
        "retry_backoff_seconds": 0.0,
    }
    values.update(updates)
    return EvaluationConfig(**values)


def _response() -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "supported": {"type": "noul", "noul": 0.82},
            "quality": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.76,
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                "legend": {"0": "weak", "1": "mixed", "2": "authoritative"},
            },
        },
        "usage": {"input_tokens": 120, "output_tokens": 8, "cost_usd": 0.001},
    }


def test_sync_provider_sends_typed_batch_and_parses_response() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = JevDecisionProvider(_config(), client=client)

    response = provider.evaluate({"report": "R", "evidence": ["E"]}, QUESTIONS)

    assert captured["authorization"] == "Bearer test-secret"
    assert captured["body"]["model"] == "jev-latest"
    assert captured["body"]["questions"]["supported"]["type"] == "noul"
    assert response.provider == "jev"
    assert response.model == "jev-1.13.0"
    assert response.answers["supported"].value == 0.82
    assert response.answers["quality"].value == 1.7
    assert response.answers["quality"].confidence == 0.76
    assert response.usage.input_tokens == 120
    assert response.usage.cost_usd == 0.001
    client.close()


def test_provider_retries_transient_status_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json=_response())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = JevDecisionProvider(_config(), client=client)

    assert provider.evaluate("state", QUESTIONS).answers["supported"].value == 0.82
    assert calls == 2
    client.close()


def test_provider_does_not_retry_non_transient_status_or_leak_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = JevDecisionProvider(_config(), client=client)

    with pytest.raises(EvaluationProviderError) as captured:
        provider.evaluate("state", QUESTIONS)

    assert calls == 1
    assert "test-secret" not in str(captured.value)
    client.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.update({"answers": {}}),
        lambda body: body["answers"]["supported"].update({"noul": 1.5}),
        lambda body: body["answers"]["quality"].update({"confidence": "high"}),
    ],
)
def test_provider_rejects_malformed_typed_answers(mutate) -> None:  # noqa: ANN001
    body = _response()
    mutate(body)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=body)
        )
    )
    provider = JevDecisionProvider(_config(), client=client)

    with pytest.raises(EvaluationResponseError):
        provider.evaluate("state", QUESTIONS)
    client.close()


def test_async_provider_uses_non_blocking_client() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response())

    async def run():  # noqa: ANN202
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = JevDecisionProvider(_config(), async_client=client)
        response = await provider.aevaluate("state", QUESTIONS)
        await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.answers["supported"].value == 0.82


def test_provider_requires_enabled_config_and_questions() -> None:
    with pytest.raises(ValueError, match="use='jev'"):
        JevDecisionProvider(EvaluationConfig())
    with pytest.raises(ValueError, match="API Key"):
        JevDecisionProvider(EvaluationConfig(use="jev"))

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_response())
        )
    )
    provider = JevDecisionProvider(_config(), client=client)
    with pytest.raises(ValueError, match="不能为空"):
        provider.evaluate("state", {})
    client.close()
