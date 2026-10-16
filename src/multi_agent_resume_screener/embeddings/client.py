"""用于简历证据检索的 embedding 客户端工厂。

检索一律使用 Gemini 的 embedding API，与 ``LLM_PROVIDER`` 无关（另一个受支持
的聊天提供方 Groq 没有 embedding API）。因此即使聊天模型本身运行在 Groq 上，
检索仍需要 ``GOOGLE_API_KEY``。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from multi_agent_resume_screener.settings import Settings, get_settings

# gemini-embedding-001：当前稳定、已正式发布（GA）、免费档的 Gemini 文本
# embedding 模型。旧教程引用的 text-embedding-004 已被弃用，Google 已于
# 2026-01-14 停用——请勿使用。
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class EmbeddingConfigError(RuntimeError):
    """当 embedding 提供方配置有误（如缺少 API key）时抛出。"""


def get_embedder(settings: Settings | None = None) -> Embeddings:
    """构建用于简历证据检索的 Gemini embeddings 客户端。

    Args:
        settings: 可选的设置覆盖（测试时有用）。缺省时回退到缓存的应用设置。

    Returns:
        一个可直接 ``.embed_documents()`` / ``.embed_query()`` 的 LangChain
        ``Embeddings`` 实例。

    Raises:
        EmbeddingConfigError: 未设置 ``GOOGLE_API_KEY`` 时抛出。
    """
    settings = settings or get_settings()
    if not settings.google_api_key:
        raise EmbeddingConfigError(
            "GOOGLE_API_KEY is not set. Evidence retrieval uses Gemini's "
            "free-tier embedding API regardless of LLM_PROVIDER; get a free "
            "key at https://aistudio.google.com/apikey and add it to your "
            ".env file."
        )
    # 延迟导入，与 llm/client.py 中的提供方导入方式保持一致。
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = settings.embedding_model or DEFAULT_EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.google_api_key)
