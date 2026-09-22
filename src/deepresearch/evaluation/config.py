"""Configuration for optional probabilistic research evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import os


EvaluationMode = Literal["shadow", "gate"]


@dataclass(frozen=True)
class EvaluationConfig:
    """Provider, payload, and bounded-decision controls for evaluation."""

    use: str = ""
    mode: EvaluationMode = "shadow"
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-latest"
    timeout_seconds: float = 15.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25
    max_payload_chars: int = 40_000
    max_report_chars: int = 20_000
    max_evidence_items: int = 12
    max_evidence_chars: int = 1_200
    max_gate_attempts: int = 1
    answer_relevance_threshold: float = 0.75
    evidence_support_threshold: float = 0.75
    citation_coverage_threshold: float = 0.70
    evidence_sufficiency_threshold: float = 0.70
    continue_research_threshold: float = 0.70
    confidence_floor: float = 0.50

    def __post_init__(self) -> None:
        if self.use not in {"", "jev"}:
            raise ValueError("DEEPRESEARCH_EVALUATION_USE 只支持空值或 jev。")
        if self.mode not in {"shadow", "gate"}:
            raise ValueError("DEEPRESEARCH_EVALUATION_MODE 必须是 shadow 或 gate。")
        if self.timeout_seconds <= 0:
            raise ValueError("DEEPRESEARCH_JEV_TIMEOUT_SECONDS 必须大于 0。")
        if self.max_retries < 0:
            raise ValueError("DEEPRESEARCH_JEV_MAX_RETRIES 不能小于 0。")
        if self.retry_backoff_seconds < 0:
            raise ValueError("DEEPRESEARCH_JEV_RETRY_BACKOFF_SECONDS 不能小于 0。")
        for name, value in (
            ("DEEPRESEARCH_EVALUATION_MAX_PAYLOAD_CHARS", self.max_payload_chars),
            ("DEEPRESEARCH_EVALUATION_MAX_REPORT_CHARS", self.max_report_chars),
            ("DEEPRESEARCH_EVALUATION_MAX_EVIDENCE_ITEMS", self.max_evidence_items),
            ("DEEPRESEARCH_EVALUATION_MAX_EVIDENCE_CHARS", self.max_evidence_chars),
        ):
            if value < 1:
                raise ValueError(f"{name} 必须大于 0。")
        if self.max_gate_attempts < 0:
            raise ValueError("DEEPRESEARCH_EVALUATION_MAX_GATE_ATTEMPTS 不能小于 0。")
        for name, value in (
            ("ANSWER_RELEVANCE", self.answer_relevance_threshold),
            ("EVIDENCE_SUPPORT", self.evidence_support_threshold),
            ("CITATION_COVERAGE", self.citation_coverage_threshold),
            ("EVIDENCE_SUFFICIENCY", self.evidence_sufficiency_threshold),
            ("CONTINUE_RESEARCH", self.continue_research_threshold),
            ("CONFIDENCE_FLOOR", self.confidence_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"DEEPRESEARCH_EVALUATION_{name}_THRESHOLD 必须在 0 到 1 之间。"
                )

    @property
    def enabled(self) -> bool:
        return bool(self.use)

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Load evaluation settings without enabling external calls by default."""

        use = os.getenv("DEEPRESEARCH_EVALUATION_USE", "").strip().lower()
        api_key = (
            os.getenv("DEEPRESEARCH_JEV_API_KEY", "").strip()
            or os.getenv("TYPESAFE_API_KEY", "").strip()
        )
        if use and not api_key:
            raise ValueError(
                "启用 Jev 评估时缺少 DEEPRESEARCH_JEV_API_KEY 或 TYPESAFE_API_KEY。"
            )
        mode = os.getenv("DEEPRESEARCH_EVALUATION_MODE", "shadow").strip().lower()
        return cls(
            use=use,
            mode=mode,  # type: ignore[arg-type]
            api_key=api_key,
            base_url=(
                os.getenv("DEEPRESEARCH_JEV_BASE_URL", "").strip()
                or "https://api.typesafe.ai/v1/systemone"
            ),
            model=(
                os.getenv("DEEPRESEARCH_JEV_MODEL", "jev-latest").strip()
                or "jev-latest"
            ),
            timeout_seconds=_env_float("DEEPRESEARCH_JEV_TIMEOUT_SECONDS", 15.0),
            max_retries=_env_int("DEEPRESEARCH_JEV_MAX_RETRIES", 1),
            retry_backoff_seconds=_env_float(
                "DEEPRESEARCH_JEV_RETRY_BACKOFF_SECONDS", 0.25
            ),
            max_payload_chars=_env_int(
                "DEEPRESEARCH_EVALUATION_MAX_PAYLOAD_CHARS", 40_000
            ),
            max_report_chars=_env_int(
                "DEEPRESEARCH_EVALUATION_MAX_REPORT_CHARS", 20_000
            ),
            max_evidence_items=_env_int(
                "DEEPRESEARCH_EVALUATION_MAX_EVIDENCE_ITEMS", 12
            ),
            max_evidence_chars=_env_int(
                "DEEPRESEARCH_EVALUATION_MAX_EVIDENCE_CHARS", 1_200
            ),
            max_gate_attempts=_env_int(
                "DEEPRESEARCH_EVALUATION_MAX_GATE_ATTEMPTS", 1
            ),
            answer_relevance_threshold=_env_float(
                "DEEPRESEARCH_EVALUATION_ANSWER_RELEVANCE_THRESHOLD", 0.75
            ),
            evidence_support_threshold=_env_float(
                "DEEPRESEARCH_EVALUATION_EVIDENCE_SUPPORT_THRESHOLD", 0.75
            ),
            citation_coverage_threshold=_env_float(
                "DEEPRESEARCH_EVALUATION_CITATION_COVERAGE_THRESHOLD", 0.70
            ),
            evidence_sufficiency_threshold=_env_float(
                "DEEPRESEARCH_EVALUATION_EVIDENCE_SUFFICIENCY_THRESHOLD", 0.70
            ),
            continue_research_threshold=_env_float(
                "DEEPRESEARCH_EVALUATION_CONTINUE_RESEARCH_THRESHOLD", 0.70
            ),
            confidence_floor=_env_float(
                "DEEPRESEARCH_EVALUATION_CONFIDENCE_FLOOR_THRESHOLD", 0.50
            ),
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数。") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
