"""Matcher 智能体：对结构化简历与结构化 JD 进行匹配评分。

这是流水线的核心，也是我们的关键差异点：matcher 不是按照固定评分标准给简历
打分，而是评估*特定*候选人与*特定*职位描述的匹配程度。它为每个板块（skills、
experience、projects、education）各生成一个 :class:`SubScore`，附引用证据和
简短理由，并输出 ``overall_evidence_quality`` 信号，供后续自我纠错环节使用。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from multi_agent_resume_screener.llm.client import get_chat_model, structured_output
from multi_agent_resume_screener.state import (
    SECTIONS,
    MatchResult,
    RetrievedChunk,
    StructuredJD,
    StructuredResume,
    SubScore,
)

SYSTEM_PROMPT = """\
You are a rigorous, fair technical hiring evaluator. You are given a candidate's \
structured resume and a structured job description. Score how well the candidate \
matches the job, section by section.

Produce exactly one entry per section: skills, experience, projects, education.

For each section:
- `score`: a float from 0.0 to 1.0 measuring fit for THIS job (not generic \
resume quality). 1.0 = excellent match; 0.0 = no relevant evidence.
- `evidence`: short snippets quoted/paraphrased from the resume that justify the \
score. If there is no relevant evidence, use an empty list.
- `reasoning`: one or two sentences explaining the score with respect to the \
job's requirements.

Also produce `overall_evidence_quality` (0.0-1.0): how strong and unambiguous \
the resume evidence was for making these judgments. Low values mean the resume \
was vague, sparse, or hard to map to the job.

Be evidence-driven and avoid inflating scores. Do not reward skills the job did \
not ask for.

If a RETRIEVED EVIDENCE section is provided below, it contains the resume \
excerpts an automated retrieval step found most relevant to each section's \
requirements. Ground your evidence and reasoning in those excerpts primarily, \
while still using the full structured resume for context. A section marked \
"no evidence retrieved above the relevance threshold" means the automated \
search found nothing strongly relevant -- it does NOT necessarily mean the \
candidate lacks the skill; use your own judgment from the full resume in that \
case, but never invent evidence that is not present in the resume.\
"""

FEEDBACK_PREFIX = """\

A previous evaluation pass was flagged as low-confidence. Address this feedback \
and re-evaluate more carefully, grounding every score in concrete resume \
evidence:
"""


def _build_human_message(
    resume: StructuredResume,
    jd: StructuredJD,
    feedback: list[str] | None,
    retrieved_evidence: dict[str, list[RetrievedChunk]] | None,
) -> str:
    parts = [
        "## JOB DESCRIPTION (structured)",
        jd.model_dump_json(indent=2),
        "",
        "## CANDIDATE RESUME (structured)",
        resume.model_dump_json(indent=2),
    ]
    if retrieved_evidence:
        parts += ["", "## RETRIEVED EVIDENCE (top-matching resume excerpts per section)"]
        for section in SECTIONS:
            chunks = retrieved_evidence.get(section) or []
            if not chunks:
                parts.append(
                    f"- {section}: (no evidence retrieved above the relevance threshold)"
                )
                continue
            parts.append(f"- {section}:")
            for rc in chunks:
                label = f" [{rc.chunk.title}]" if rc.chunk.title else ""
                parts.append(f"  - (score={rc.score:.2f}){label} {rc.chunk.text}")
    if feedback:
        parts.append(FEEDBACK_PREFIX)
        parts.extend(f"- {item}" for item in feedback)
    return "\n".join(parts)


def _normalise_sections(result: MatchResult) -> MatchResult:
    """确保每个已知板块恰好有一个子分数。

    虽然已要求 LLM 每个板块只输出一条，我们仍会防御性地去重（每板块保留
    第一条），并为缺失的板块填充 0 分，确保下游评分总是拿到完整、可预测的
    集合。
    """
    by_section: dict[str, SubScore] = {}
    for sub in result.sub_scores:
        if sub.section in SECTIONS and sub.section not in by_section:
            by_section[sub.section] = sub

    normalised = [
        by_section.get(
            section,
            SubScore(
                section=section,
                score=0.0,
                evidence=[],
                reasoning="No relevant evidence found.",
            ),
        )
        for section in SECTIONS
    ]
    return MatchResult(
        sub_scores=normalised,
        overall_evidence_quality=result.overall_evidence_quality,
    )


def match(
    resume: StructuredResume,
    jd: StructuredJD,
    llm: BaseChatModel | None = None,
    feedback: list[str] | None = None,
    retrieved_evidence: dict[str, list[RetrievedChunk]] | None = None,
) -> MatchResult:
    """根据职位描述评估一份简历。

    Args:
        resume: 结构化简历。
        jd: 结构化职位描述。
        llm: 可选的聊天模型（用于测试的依赖注入）。缺省时通过
            :func:`get_chat_model` 回退到已配置的提供方。
        feedback: 可选的上一轮 critic 建议，注入提示词以驱动唯一一次自我
            纠错复评。
        retrieved_evidence: 可选的每个板块 top-k 简历摘录，来自证据检索层，
            用于为 matcher 的评分提供依据。

    Returns:
        一个 :class:`MatchResult`，每个板块恰好一个 :class:`SubScore`，并带
        ``overall_evidence_quality`` 信号。
    """
    llm = llm or get_chat_model()
    matcher = structured_output(llm, MatchResult)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=_build_human_message(resume, jd, feedback, retrieved_evidence)
        ),
    ]
    result = matcher.invoke(messages)

    if isinstance(result, dict):
        result = MatchResult.model_validate(result)
    return _normalise_sections(result)
