"""对所配置的 LLM 提供商做快速连通性检查。

用法（在项目根目录、已激活虚拟环境时）：

    python scripts/check_llm.py

从 .env 文件读取配置，通过与提供商无关的工厂构建聊天模型，
发送一个极小的测试 prompt 并打印回复。绝不打印你的 API key。
出现任何失败都以非零码退出，因此可用于 CI。
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from multi_agent_resume_screener.llm.client import LLMConfigError, get_chat_model
from multi_agent_resume_screener.settings import get_settings


def main() -> int:
    # 确保 .env 已加载，即使进程没有自动加载它。
    load_dotenv()

    settings = get_settings()
    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.llm_model or '(provider default)'}")

    try:
        model = get_chat_model()
    except LLMConfigError as exc:
        print(f"\n[CONFIG ERROR] {exc}")
        return 2

    prompt = "Reply with exactly the word: pong"
    print(f"\nSending test prompt: {prompt!r}")

    try:
        response = model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - 直接抛出任何提供商/网络错误
        print(f"\n[REQUEST FAILED] {type(exc).__name__}: {exc}")
        print("Check that your API key is valid and you have network access.")
        return 1

    text = getattr(response, "content", str(response))
    print(f"\nLLM replied: {text!r}")
    print("\nOK - LLM client is working end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
