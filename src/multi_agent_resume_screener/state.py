"""智能体之间共享的流水线状态与类型化契约。

每个 LangGraph 节点都会收到一个 :class:`PipelineState`，填入自己负责的部分，
再传递下去。使用显式的 Pydantic 模型（而非松散的字典）意味着 LangGraph 会
校验在智能体之间流动的数据，整条流水线也因此自文档化。

一个 :class:`PipelineState` 代表一份简历在流水线中的完整旅程。招聘者模式下
API 会为每份简历运行一个状态（并行执行），随后对得到的 :class:`CandidateResult`
对象进行排序。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# 简历评分所依据的各板块，及其对最终分数的默认贡献。
# 权重只是一个起点，可按请求覆盖，且始终会归一化到总和为 1.0
# （见 PipelineConfig）。
Section = Literal["skills", "experience", "projects", "education"]
SECTIONS: tuple[Section, ...] = ("skills", "experience", "projects", "education")
DEFAULT_WEIGHTS: dict[str, float] = {
    "skills": 0.35,
    "experience": 0.30,
    "projects": 0.25,
    "education": 0.10,
}

Mode = Literal["candidate", "recruiter"]
CriticMode = Literal["fast", "full"]
Verdict = Literal["strong_fit", "moderate_fit", "weak_fit"]


def _now_iso() -> str:
    """以 ISO-8601 字符串返回当前 UTC 时间（用于审计记录）。"""
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #
class ResumeRaw(BaseModel):
    """经 PDF -> 文本提取（确定性函数）之后的一份简历。"""

    filename: str
    text: str


class JDRaw(BaseModel):
    """原始职位描述文本。"""

    text: str


# --------------------------------------------------------------------------- #
# 解析器输出（结构化简历 / JD）
# --------------------------------------------------------------------------- #
class EducationItem(BaseModel):
    degree: str | None = None
    institute: str | None = None
    year: str | None = None
    gpa: str | None = None


class ExperienceItem(BaseModel):
    company: str | None = None
    role: str | None = None
    dates: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str | None = None
    description: str | None = None
    tech: list[str] = Field(default_factory=list)


class StructuredResume(BaseModel):
    """由 parser 智能体从简历中提取的结构化字段。"""

    name: str | None = None
    email: str | None = None
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class StructuredJD(BaseModel):
    """从职位描述中提取的结构化要求。"""

    title: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_experience_years: float | None = None
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 检索（证据分块 + 为 matcher 检索到的内容）
# --------------------------------------------------------------------------- #
class ResumeChunk(BaseModel):
    """结构化简历中一个可检索单元。

    每条经历/项目/教育条目各一个分块，整份（扁平的）技能列表也单独一个
    分块——见 ``pipeline/retrieval.py::build_chunks``。
    """

    section: Section
    title: str | None = None
    text: str


class RetrievedChunk(BaseModel):
    """为某个板块基于 JD 生成的查询所检索到的简历分块。"""

    chunk: ResumeChunk
    score: float = Field(ge=-1.0, le=1.0)


# --------------------------------------------------------------------------- #
# matcher 输出
# --------------------------------------------------------------------------- #
class SubScore(BaseModel):
    """每个板块的匹配分数，以及支撑该分数的证据。"""

    section: Section
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str = ""


class MatchResult(BaseModel):
    """所有板块的子分数，外加一个整体证据质量信号。

    ``overall_evidence_quality``（0-1）用于自我纠错决策：如果 matcher 自身的
    证据较弱，critic 就更可能要求重新评估。
    """

    sub_scores: list[SubScore] = Field(default_factory=list)
    overall_evidence_quality: float = Field(default=0.0, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# 评分器输出（确定性函数，不是智能体）
# --------------------------------------------------------------------------- #
class FinalScore(BaseModel):
    """由 matcher 的子分数推导出的确定性加权分数。"""

    score: float = Field(ge=0.0, le=1.0)
    weights_used: dict[str, float] = Field(default_factory=dict)
    breakdown: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# critic 输出
# --------------------------------------------------------------------------- #
class Critique(BaseModel):
    """critic 的反馈及其对评分结果的置信度。

    ``needs_rescore`` 与重试上限共同驱动那条回到 matcher 的唯一
    自我纠错回路。
    """

    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    verdict: Verdict = "moderate_fit"
    confidence_in_scoring: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_rescore: bool = False


# --------------------------------------------------------------------------- #
# 简历体检报告（确定性、客观的简历检查）
# --------------------------------------------------------------------------- #
class HygieneIssue(BaseModel):
    """确定性简历体检检查发现的一个客观问题。"""

    check: str
    severity: Literal["info", "warning"]
    message: str


class HygieneReport(BaseModel):
    """独立于 JD 匹配度的客观简历质量信号。

    这只是建议性的：它为 critic 的建议提供参考并展示给候选人，但不会改变
    JD 匹配的 ``FinalScore``。
    """

    score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[HygieneIssue] = Field(default_factory=list)
    positives: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
class PipelineConfig(BaseModel):
    """按请求的配置。

    - ``mode`` 选择角色：``candidate``（简历反馈）或 ``recruiter``
      （排序筛选）。
    - ``critic_mode`` 控制成本：``full`` 评估每份简历；``fast`` 只在排序后
      评估前 ``critic_top_k`` 名（招聘者模式）。
    - ``weights`` 会被归一化到总和为 1.0，从而保证最终分数始终落在 [0, 1]。
    """

    mode: Mode = "recruiter"
    critic_mode: CriticMode = "full"
    critic_top_k: int = Field(default=5, ge=1)
    enable_critic: bool = True
    retrieval_top_k: int = Field(default=4, ge=1)
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_retries: int = Field(default=1, ge=0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalise_weights(self) -> PipelineConfig:
        # 从默认值开始，只应用有效的板块覆盖项，丢弃未知键。
        merged = dict(DEFAULT_WEIGHTS)
        for key, value in self.weights.items():
            if key in SECTIONS and value >= 0:
                merged[key] = float(value)

        total = sum(merged.values())
        if total <= 0:
            merged = dict(DEFAULT_WEIGHTS)
            total = sum(merged.values())

        # 归一化，使权重之和始终恰好为 1.0。
        self.weights = {k: v / total for k, v in merged.items()}
        return self


# --------------------------------------------------------------------------- #
# 审计轨迹
# --------------------------------------------------------------------------- #
class TraceEntry(BaseModel):
    """流水线审计日志中的一步（可解释性的基石）。"""

    agent: str
    timestamp: str = Field(default_factory=_now_iso)
    note: str = ""
    tokens_used: int | None = None


# --------------------------------------------------------------------------- #
# 共享状态（每份简历一个）
# --------------------------------------------------------------------------- #
class PipelineState(BaseModel):
    """在 LangGraph 流水线中随单份简历传递的可变状态。"""

    # 输入
    resume_raw: ResumeRaw
    jd_raw: JDRaw
    config: PipelineConfig = Field(default_factory=PipelineConfig)

    # 随各智能体运行逐步累积
    resume_structured: StructuredResume | None = None
    jd_structured: StructuredJD | None = None
    resume_chunks: list[ResumeChunk] = Field(default_factory=list)
    retrieved_evidence: dict[str, list[RetrievedChunk]] = Field(default_factory=dict)
    match_result: MatchResult | None = None
    final_score: FinalScore | None = None
    hygiene: HygieneReport | None = None
    critique: Critique | None = None

    # 自我纠错记录（上限见 config.max_retries）
    retry_count: int = 0

    # 可解释性 / 调试
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def add_trace(
        self, agent: str, note: str = "", tokens_used: int | None = None
    ) -> None:
        """为一次智能体调用追加一条审计日志。"""
        self.trace.append(
            TraceEntry(agent=agent, note=note, tokens_used=tokens_used)
        )

    def can_retry(self) -> bool:
        """如果还允许进行一次自我纠错则返回 True。"""
        return self.retry_count < self.config.max_retries


# --------------------------------------------------------------------------- #
# 对外输出（与内部状态解耦）
# --------------------------------------------------------------------------- #
class CandidateResult(BaseModel):
    """单份简历干净、面向 API 的结果。

    与 :class:`PipelineState` 分开维护，以免外部契约泄露内部簿记信息
    （重试计数、原始文本等）。
    """

    filename: str
    candidate_name: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    breakdown: dict[str, float] = Field(default_factory=dict)

    # "长期积累" 类：JD 中未被明确满足的要求（``gaps``），以及 critic 给出的
    # 面向角色定位的能力建设建议（``suggestions``）。
    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    # "立刻可修" 类：客观、确定性的简历质量信号，以及它发现的具体快速
    # 修改项（``hygiene_issues``）。
    hygiene_score: float | None = None
    hygiene_issues: list[str] = Field(default_factory=list)

    # 候选人为何排在当前位置：matcher 的分板块推理与支撑证据，为可解释性
    # 而展示（"展示排序理由"）。按板块（skills/experience/projects/education）
    # 作为键。
    section_reasons: dict[str, str] = Field(default_factory=dict)
    section_evidence: dict[str, list[str]] = Field(default_factory=dict)

    # 支撑 matcher 判断的简历检索摘录，按板块作为键
    # （RAG 证据检索层对审计轨迹的贡献）。
    retrieved_evidence: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_state(cls, state: PipelineState) -> CandidateResult:
        """从处理完毕的流水线状态构建对外的结果。"""
        score = state.final_score.score if state.final_score else 0.0
        breakdown = state.final_score.breakdown if state.final_score else {}
        verdict: Verdict = state.critique.verdict if state.critique else "weak_fit"
        gaps = state.critique.gaps if state.critique else []
        suggestions = state.critique.suggestions if state.critique else []
        name = state.resume_structured.name if state.resume_structured else None
        hygiene_score = state.hygiene.score if state.hygiene else None
        hygiene_issues = (
            [issue.message for issue in state.hygiene.issues]
            if state.hygiene
            else []
        )
        # 把 matcher 的分板块推理/证据作为"排序理由"展示
        # （若流水线在匹配之前就失败则为空）。
        section_reasons: dict[str, str] = {}
        section_evidence: dict[str, list[str]] = {}
        if state.match_result is not None:
            for sub in state.match_result.sub_scores:
                if sub.reasoning:
                    section_reasons[sub.section] = sub.reasoning
                if sub.evidence:
                    section_evidence[sub.section] = list(sub.evidence)
        # 检索到的证据，展平为引用文本（省略空板块，
        # 与上面 section_evidence 省略空的约定保持一致）。
        retrieved_evidence: dict[str, list[str]] = {
            section: [rc.chunk.text for rc in chunks]
            for section, chunks in state.retrieved_evidence.items()
            if chunks
        }
        return cls(
            filename=state.resume_raw.filename,
            candidate_name=name,
            score=score,
            verdict=verdict,
            breakdown=breakdown,
            gaps=gaps,
            suggestions=suggestions,
            hygiene_score=hygiene_score,
            hygiene_issues=hygiene_issues,
            section_reasons=section_reasons,
            section_evidence=section_evidence,
            retrieved_evidence=retrieved_evidence,
        )


class ScreeningResult(BaseModel):
    """一次筛选请求的完整响应（一名或多名候选人）。"""

    run_id: str | None = None
    created_at: str | None = None
    mode: Mode
    job_title: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)

    @property
    def top(self) -> CandidateResult | None:
        """得分最高的候选人（如有）。"""
        return self.candidates[0] if self.candidates else None
