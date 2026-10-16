"""Parser 智能体：简历文本 -> :class:`StructuredResume`。

这是第一个真正的*智能体*。从自由格式的简历文本中提取结构化字段需要判断力
（章节边界、什么算"技能"、如何拆分要点），因此我们使用 LLM。我们依赖
LangChain 的 ``with_structured_output``，让模型直接返回经过校验的 Pydantic
对象，而不是需要我们自行解析和修复的自由文本。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from multi_agent_resume_screener.llm.client import get_chat_model, structured_output
from multi_agent_resume_screener.state import ResumeRaw, StructuredResume

SYSTEM_PROMPT = """\
You are a meticulous resume parser. Extract the candidate's information from the \
resume text into the required structured format.

Rules:
- Extract only what is present. Never invent or infer missing information; leave \
a field empty or null if the resume does not state it.
- `skills` must be a flat list of individual skills/technologies (split comma- \
or bullet-separated lists into separate items; do not include whole sentences).
- For `experience`, capture each role's company, title, dates, and its bullet \
points verbatim (lightly cleaned of formatting artefacts).
- For `projects`, capture the name, a short description, and the technologies \
used.
- For `education`, capture degree, institute, year, and GPA/percentage if shown.
- Preserve the candidate's wording; do not summarise or embellish.\
"""


def parse_resume(
    resume_raw: ResumeRaw,
    llm: BaseChatModel | None = None,
) -> StructuredResume:
    """将原始简历文本解析为 :class:`StructuredResume`。

    Args:
        resume_raw: 简历文件名 + 提取出的文本。
        llm: 可选的聊天模型（用于测试的依赖注入）。缺省时通过
            :func:`get_chat_model` 回退到已配置的提供方。

    Returns:
        填充完成的 :class:`StructuredResume`。若输入为空，则直接返回空的
        ``StructuredResume``，不调用 LLM。
    """
    if not resume_raw.text.strip():
        return StructuredResume()

    llm = llm or get_chat_model()
    extractor = structured_output(llm, StructuredResume)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=resume_raw.text),
    ]
    result = extractor.invoke(messages)

    # 某些提供方可能返回 dict 而不是模型实例。
    if isinstance(result, dict):
        result = StructuredResume.model_validate(result)
    return result
