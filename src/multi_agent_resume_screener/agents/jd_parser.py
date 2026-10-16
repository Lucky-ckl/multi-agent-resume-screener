"""JD parser 智能体：职位描述文本 -> :class:`StructuredJD`。

与简历解析器对应：从自由格式的职位描述中提取结构化要求需要判断力（哪些是
*必需*技能、哪些是*加分项*、隐含多少年经验），因此我们使用带结构化输出的
LLM。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from multi_agent_resume_screener.llm.client import get_chat_model, structured_output
from multi_agent_resume_screener.state import JDRaw, StructuredJD

SYSTEM_PROMPT = """\
You are an expert technical recruiter. Extract the hiring requirements from the \
job description into the required structured format.

Rules:
- Extract only what the job description states; do not invent requirements.
- `required_skills`: skills/technologies the candidate MUST have (listed as \
"required", "must have", or clearly essential). Use a flat list of individual \
items.
- `nice_to_have_skills`: skills described as "preferred", "bonus", "a plus", or \
"nice to have". Do not duplicate items already in `required_skills`.
- `min_experience_years`: the minimum years of experience as a number if stated \
(e.g. "3+ years" -> 3). Leave null if not specified.
- `responsibilities`: what the person will do in the role.
- `qualifications`: education, certifications, or background requirements.
- Keep each list item concise and specific.\
"""


def parse_jd(
    jd_raw: JDRaw,
    llm: BaseChatModel | None = None,
) -> StructuredJD:
    """将原始职位描述解析为 :class:`StructuredJD`。

    Args:
        jd_raw: 原始职位描述文本。
        llm: 可选的聊天模型（用于测试的依赖注入）。缺省时通过
            :func:`get_chat_model` 回退到已配置的提供方。

    Returns:
        填充完成的 :class:`StructuredJD`。若输入为空，则直接返回空的
        ``StructuredJD``，不调用 LLM。
    """
    if not jd_raw.text.strip():
        return StructuredJD()

    llm = llm or get_chat_model()
    extractor = structured_output(llm, StructuredJD)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=jd_raw.text),
    ]
    result = extractor.invoke(messages)

    if isinstance(result, dict):
        result = StructuredJD.model_validate(result)
    return result
