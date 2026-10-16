"""应用配置，从环境变量 / .env 文件加载。

所有配置集中在此，保证单一事实来源。密钥（API key）只从环境读取，
绝不硬编码。可用变量见 `.env.example`。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境读取的强类型应用设置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM 提供商选择 ---
    # 使用哪个提供商："gemini"（默认）或 "groq"。
    llm_provider: str = Field(default="gemini")
    # 可选的显式模型名。若未设置，则使用各提供商的默认模型。
    llm_model: str | None = Field(default=None)

    # --- 嵌入模型（简历证据检索）---
    # 无论 LLM_PROVIDER 取值如何，检索始终使用 Gemini 的嵌入 API
    # （Groq 没有嵌入 API），因此该项需要 GOOGLE_API_KEY。
    # 可选的显式模型名；未设置时默认为 "gemini-embedding-001"。
    embedding_model: str | None = Field(default=None)

    # --- 提供商 API 密钥 ---
    google_api_key: str | None = Field(default=None)
    groq_api_key: str | None = Field(default=None)

    # --- 应用 ---
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    db_path: str = Field(default="runs.db")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的 Settings 实例（.env 文件只读取一次）。"""
    return Settings()
