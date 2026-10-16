"""Critic 智能体：审查匹配结果、生成反馈，并控制自我纠错。

critic 是流水线中的最后一个智能体。它会查看结构化简历、职位描述、matcher 的
各板块分数与证据、最终得分以及确定性的规范报告，并产出：

- ``gaps``：候选人未明确满足的具体 JD 要求
- ``suggestions``：可落地的改进建议（含规范修复）
- ``verdict``：总体 strong/moderate/weak 匹配标签
- ``confidence_in_scoring``：matcher 各项评分的论证充分程度

是否回环进行唯一一次复评由置信度与 matcher 的证据质量*确定性*决定（不交给
LLM），因此自我纠错行为可预测、可调节。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from multi_agent_resume_screener.llm.client import get_chat_model, structured_output
from multi_agent_resume_screener.state import (
    Critique,
    FinalScore,
    HygieneReport,
    MatchResult,
    PipelineConfig,
    StructuredJD,
    StructuredResume,
    Verdict,
)


class _CriticLLMOutput(BaseModel):
    """我们要求 LLM 输出的内容。``needs_rescore`` 会在之后确定性地决定，
    因此特意不纳入此模式。"""

    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    verdict: Verdict = "moderate_fit"
    confidence_in_scoring: float = Field(default=1.0, ge=0.0, le=1.0)


SYSTEM_PROMPT = """\
You are a constructive hiring reviewer. You are given a job description, a \
candidate's structured resume, the per-section match scores with their evidence, \
the final weighted score, and a deterministic resume-hygiene report.

Produce:
- `gaps`: specific requirements from the JOB DESCRIPTION that the candidate does \
not clearly satisfy. Be concrete (name the missing skill/experience). Empty list \
if there are none.
- `suggestions`: forward-looking, skill-building advice the candidate should work \
on OVER TIME to become a stronger fit for THIS role — concrete skills to learn, \
technologies to gain depth in, and the kind of experience or projects to build. \
Frame each as something to develop next (e.g. "Build a service using Kafka to \
show streaming experience"). Do NOT include cosmetic resume-editing fixes such as \
bolding, adding numbers, shortening bullets, or adding links — a separate \
deterministic checker already handles those immediate edits, so avoid duplicating \
them here. Keep each suggestion short and specific.
- `verdict`: one of strong_fit, moderate_fit, weak_fit, reflecting overall fit.
- `confidence_in_scoring`: 0.0-1.0 — how confident you are that the per-section \
scores are well-justified by the cited evidence. Lower this when evidence is \
thin, ambiguous, or seems inconsistent with the scores.

Be honest and specific; do not pad the lists. Use the hygiene report only as \
context on the resume's current state, not as a source of suggestions.\
"""


def _build_human_message(
    resume: StructuredResume,
    jd: StructuredJD,
    match_result: MatchResult,
    final_score: FinalScore,
    hygiene: HygieneReport | None,
) -> str:
    lines = [
        "## JOB DESCRIPTION",
        jd.model_dump_json(indent=2),
        "",
        "## CANDIDATE RESUME",
        resume.model_dump_json(indent=2),
        "",
        "## PER-SECTION MATCH SCORES",
    ]
    for sub in match_result.sub_scores:
        lines.append(
            f"- {sub.section}: {sub.score:.2f} | evidence={sub.evidence} | "
            f"reasoning={sub.reasoning}"
        )
    lines.append(f"\nOverall evidence quality: {match_result.overall_evidence_quality:.2f}")
    lines.append(f"Final weighted score: {final_score.score:.2f}")

    if hygiene is not None:
        lines.append("\n## RESUME HYGIENE (deterministic)")
        lines.append(f"Hygiene score: {hygiene.score:.2f}")
        for issue in hygiene.issues:
            lines.append(f"- [{issue.severity}] {issue.message}")
    return "\n".join(lines)


def critique(
    resume: StructuredResume,
    jd: StructuredJD,
    match_result: MatchResult,
    final_score: FinalScore,
    config: PipelineConfig,
    hygiene: HygieneReport | None = None,
    llm: BaseChatModel | None = None,
) -> Critique:
    """生成 critic 反馈，并决定是否进行自我纠错。

    Args:
        resume: 结构化简历。
        jd: 结构化职位描述。
        match_result: matcher 输出（子分数 + 证据质量）。
        final_score: 确定性最终得分。
        config: 流水线配置（提供置信度阈值）。
        hygiene: 可选的确定性规范报告，用于并入改进建议。
        llm: 可选的聊天模型（用于测试的依赖注入）。

    Returns:
        一个 :class:`Critique`。``needs_rescore`` 为确定性设置：当 critic 的
        置信度或 matcher 的证据质量低于 ``config.confidence_threshold`` 时为
        true。
    """
    llm = llm or get_chat_model()
    reviewer = structured_output(llm, _CriticLLMOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=_build_human_message(
                resume, jd, match_result, final_score, hygiene
            )
        ),
    ]
    out = reviewer.invoke(messages)
    if isinstance(out, dict):
        out = _CriticLLMOutput.model_validate(out)

    # 确定性的自我纠错决策（不交给 LLM 决定）。
    threshold = config.confidence_threshold
    needs_rescore = (
        out.confidence_in_scoring < threshold
        or match_result.overall_evidence_quality < threshold
    )

    return Critique(
        gaps=out.gaps,
        suggestions=out.suggestions,
        verdict=out.verdict,
        confidence_in_scoring=out.confidence_in_scoring,
        needs_rescore=needs_rescore,
    )
