"""与提供方无关的 LLM 客户端工厂。

各智能体只依赖 LangChain 的 :class:`BaseChatModel` 抽象，而不依赖任何具体的
提供方类。因此切换提供方（如 Gemini -> Groq）只需改配置（``.env`` 中的
``LLM_PROVIDER``），智能体代码本身无需改动。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from multi_agent_resume_screener.settings import Settings, get_settings

# 各提供方合理的免费档默认模型，在未设置 LLM_MODEL 时使用。
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-20b",
}


class LLMConfigError(RuntimeError):
    """当 LLM 提供方配置有误（如缺少 API key）时抛出。"""


def get_chat_model(
    *,
    temperature: float = 0.0,
    settings: Settings | None = None,
    **kwargs,
) -> BaseChatModel:
    """按 settings 中指定的提供方构建一个配置好的聊天模型。

    Args:
        temperature: 采样温度。默认 0.0，以保证提取和评分结果确定、可复现。
        settings: 可选的设置覆盖（测试时有用）。缺省时回退到缓存的应用设置。
        **kwargs: 额外的提供方专属关键字参数，透传给底层 LangChain 聊天模型。

    Returns:
        一个可直接 ``.invoke()`` 或用 ``.with_structured_output()`` 包装的
        LangChain ``BaseChatModel``。

    Raises:
        LLMConfigError: 提供方不受支持或其 API key 缺失时抛出。
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider not in DEFAULT_MODELS:
        raise LLMConfigError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Supported: {', '.join(DEFAULT_MODELS)}."
        )

    model = settings.llm_model or DEFAULT_MODELS[provider]

    if provider == "gemini":
        return _build_gemini(model, temperature, settings, **kwargs)
    if provider == "groq":
        return _build_groq(model, temperature, settings, **kwargs)

    # 防御性代码：有上面的成员检查，理论上不应到达此处。
    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{provider}'. "
        f"Supported: {', '.join(DEFAULT_MODELS)}."
    )


def structured_output(llm: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    """把聊天模型包装为输出 ``schema`` 实例的结构化链。

    优先使用 ``json_schema`` 模式：Groq 等提供方会在服务端按 schema 做
    约束解码，从根本上避免模型漏调工具或产出不合 schema 的 JSON（部分
    模型在 ``tool_choice=required`` 下仍会直接输出纯文本）。提供方不支持
    该模式时（如当前版本的 Gemini 客户端），自动回退到默认的
    ``function_calling`` 模式。
    """
    try:
        return llm.with_structured_output(schema, method="json_schema")
    except (TypeError, ValueError):
        return llm.with_structured_output(schema)


def _build_gemini(
    model: str, temperature: float, settings: Settings, **kwargs
) -> BaseChatModel:
    if not settings.google_api_key:
        raise LLMConfigError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )
    # 延迟导入，纯 Groq 部署便无需安装此包。
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        **kwargs,
    )


def _build_groq(
    model: str, temperature: float, settings: Settings, **kwargs
) -> BaseChatModel:
    if not settings.groq_api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add it to your .env file."
        )
    # 延迟导入；langchain-groq 是可选依赖。
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMConfigError(
            "langchain-groq is not installed. Run "
            "`pip install langchain-groq` to use the Groq provider."
        ) from exc

    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        # 免费档 TPM 限额较低（如 gpt-oss-120b 每分钟 8000 tokens），一次
        # 完整流水线可能触发 429。SDK 会按 Retry-After 自动退避重试，
        # 加大重试次数让流水线能自行熬过限速窗口。
        max_retries=4,
        **kwargs,
    )
