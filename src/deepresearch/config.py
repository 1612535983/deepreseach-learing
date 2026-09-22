"""Read the minimal model configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from deepresearch.memory.config import MemoryConfig


@dataclass(frozen=True)
class Settings:
    """Configuration needed to construct one OpenAI-compatible chat model."""

    api_key: str
    model: str = "deepseek-chat"
    base_url: str | None = "https://api.deepseek.com"
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("DEEPRESEARCH_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "缺少 DEEPRESEARCH_API_KEY。请复制 .env.example 为 .env，"
                "并填写模型服务的 API Key。"
            )

        base_url = os.getenv("DEEPRESEARCH_BASE_URL", "").strip() or None
        model = os.getenv("DEEPRESEARCH_MODEL", "deepseek-chat").strip()
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            memory=MemoryConfig.from_env(),
        )
