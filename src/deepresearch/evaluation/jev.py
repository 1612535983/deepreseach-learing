"""Strict sync/async adapter for TypeSafe AI's Jev decision endpoint."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import httpx

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.exceptions import (
    EvaluationProviderError,
    EvaluationResponseError,
)
from deepresearch.evaluation.types import (
    DecisionAnswer,
    DecisionQuestion,
    DecisionResponse,
    DecisionUsage,
)


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class JevDecisionProvider:
    """Call Jev while keeping vendor wire formats outside Agent code."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.use != "jev":
            raise ValueError("JevDecisionProvider 需要 use='jev'。")
        if not config.api_key:
            raise ValueError("JevDecisionProvider 缺少 API Key。")
        self._config = config
        self._client = client
        self._async_client = async_client

    @property
    def name(self) -> str:
        return "jev"

    def evaluate(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionResponse:
        payload = self._payload(state, questions)
        started = time.perf_counter()
        owned = self._client is None
        context = (
            httpx.Client(timeout=self._config.timeout_seconds)
            if owned
            else nullcontext(self._client)
        )
        try:
            with context as client:
                if client is None:
                    raise EvaluationProviderError("Jev 同步 Client 初始化失败。")
                response = self._send_with_retries(client, payload)
        except EvaluationProviderError:
            raise
        except httpx.HTTPError as exc:
            raise EvaluationProviderError(
                f"Jev 请求失败：{type(exc).__name__}"
            ) from exc
        latency_ms = max(0, round((time.perf_counter() - started) * 1_000))
        return self._parse(response, questions, latency_ms)

    async def aevaluate(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionResponse:
        payload = self._payload(state, questions)
        started = time.perf_counter()
        if self._async_client is not None:
            response = await self._asend_with_retries(self._async_client, payload)
        else:
            try:
                async with httpx.AsyncClient(
                    timeout=self._config.timeout_seconds
                ) as client:
                    response = await self._asend_with_retries(client, payload)
            except EvaluationProviderError:
                raise
            except httpx.HTTPError as exc:
                raise EvaluationProviderError(
                    f"Jev 请求失败：{type(exc).__name__}"
                ) from exc
        latency_ms = max(0, round((time.perf_counter() - started) * 1_000))
        return self._parse(response, questions, latency_ms)

    def _send_with_retries(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
    ) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = client.post(
                    self._config.base_url,
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._raise_for_status(response)
                    return response
                last_error = httpx.HTTPStatusError(
                    f"Jev returned retryable HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            if attempt < self._config.max_retries:
                time.sleep(self._config.retry_backoff_seconds * (attempt + 1))
        raise EvaluationProviderError(
            f"Jev 请求重试后仍失败：{type(last_error).__name__}"
        ) from last_error

    async def _asend_with_retries(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = await client.post(
                    self._config.base_url,
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._raise_for_status(response)
                    return response
                last_error = httpx.HTTPStatusError(
                    f"Jev returned retryable HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            if attempt < self._config.max_retries:
                await asyncio.sleep(
                    self._config.retry_backoff_seconds * (attempt + 1)
                )
        raise EvaluationProviderError(
            f"Jev 请求重试后仍失败：{type(last_error).__name__}"
        ) from last_error

    def _payload(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("Jev questions 不能为空。")
        return {
            "state": dict(state) if isinstance(state, Mapping) else state,
            "model": self._config.model,
            "questions": {
                question_id: question.to_payload()
                for question_id, question in questions.items()
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EvaluationProviderError(
                f"Jev 返回 HTTP {response.status_code}。"
            ) from exc

    def _parse(
        self,
        response: httpx.Response,
        questions: Mapping[str, DecisionQuestion],
        latency_ms: int,
    ) -> DecisionResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise EvaluationResponseError("Jev 返回的不是有效 JSON。") from exc
        if not isinstance(body, Mapping):
            raise EvaluationResponseError("Jev 响应根节点必须是 object。")
        raw_answers = body.get("answers")
        if not isinstance(raw_answers, Mapping):
            raise EvaluationResponseError("Jev 响应缺少 answers object。")
        missing = set(questions).difference(raw_answers)
        if missing:
            raise EvaluationResponseError(
                "Jev 响应缺少问题：" + ", ".join(sorted(missing))
            )
        answers = {
            question_id: self._parse_answer(
                question_id,
                raw_answers[question_id],
                question,
            )
            for question_id, question in questions.items()
        }
        raw_usage = body.get("usage")
        usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        model = body.get("model")
        return DecisionResponse(
            provider=self.name,
            model=str(model or self._config.model),
            answers=answers,
            usage=DecisionUsage(
                input_tokens=self._non_negative_int(usage.get("input_tokens")),
                output_tokens=self._non_negative_int(usage.get("output_tokens")),
                cost_usd=self._optional_non_negative_float(
                    usage.get("cost_usd") or usage.get("cost")
                ),
            ),
            latency_ms=latency_ms,
        )

    def _parse_answer(
        self,
        question_id: str,
        raw: object,
        question: DecisionQuestion,
    ) -> DecisionAnswer:
        if not isinstance(raw, Mapping):
            raise EvaluationResponseError(
                f"Jev answer {question_id} 必须是 object。"
            )
        answer_type = str(raw.get("type") or "")
        if answer_type != question.question_type:
            raise EvaluationResponseError(
                f"Jev answer {question_id} 类型不匹配：{answer_type}。"
            )
        if answer_type == "noul":
            value: str | float = self._probability(raw.get("noul"), question_id)
        elif answer_type == "choice":
            value = str(raw.get("choice") or "")
            if not value:
                raise EvaluationResponseError(
                    f"Jev answer {question_id} 缺少 choice。"
                )
            if isinstance(question.criteria, dict) and value not in question.criteria:
                raise EvaluationResponseError(
                    f"Jev answer {question_id} 返回未知 choice：{value}。"
                )
        else:
            value = self._finite_float(raw.get("score"), question_id)

        probabilities = self._probabilities(raw.get("probabilities"), question_id)
        confidence = None
        if answer_type in {"choice", "score"}:
            confidence = self._probability(raw.get("confidence"), question_id)
            if not probabilities:
                raise EvaluationResponseError(
                    f"Jev answer {question_id} 缺少 probabilities。"
                )
        raw_legend = raw.get("legend")
        legend = (
            {str(key): str(item) for key, item in raw_legend.items()}
            if isinstance(raw_legend, Mapping)
            else {}
        )
        return DecisionAnswer(
            answer_type=answer_type,  # type: ignore[arg-type]
            value=value,
            probabilities=probabilities,
            confidence=confidence,
            legend=legend,
        )

    def _probabilities(self, raw: object, question_id: str) -> dict[str, float]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise EvaluationResponseError(
                f"Jev answer {question_id} probabilities 必须是 object。"
            )
        return {
            str(key): self._probability(value, question_id)
            for key, value in raw.items()
        }

    @staticmethod
    def _finite_float(raw: object, question_id: str) -> float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EvaluationResponseError(
                f"Jev answer {question_id} 必须返回有限数字。"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise EvaluationResponseError(
                f"Jev answer {question_id} 必须返回有限数字。"
            )
        return value

    def _probability(self, raw: object, question_id: str) -> float:
        value = self._finite_float(raw, question_id)
        if not 0.0 <= value <= 1.0:
            raise EvaluationResponseError(
                f"Jev answer {question_id} 概率必须在 0 到 1 之间。"
            )
        return value

    @staticmethod
    def _non_negative_int(raw: object) -> int:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return 0
        return max(0, int(raw))

    @staticmethod
    def _optional_non_negative_float(raw: object) -> float | None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        return value if math.isfinite(value) and value >= 0 else None
