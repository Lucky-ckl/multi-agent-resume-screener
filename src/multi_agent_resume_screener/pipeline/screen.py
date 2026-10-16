"""位于单简历流水线之上的高层筛选编排。

该层处于 LangGraph 流水线之上，负责处理两种角色：

- ``candidate`` 模式：一份简历，获得完整的 critic 反馈。
- ``recruiter`` 模式：多份简历针对同一个 JD 排序。``full`` critic 模式下
  每位候选人都会被评估；``fast`` 模式下只评估排序后的前 K 名
  （在大规模场景下节省 LLM 调用）。

JD 只解析一次并在所有简历间复用（流水线的 parser 节点是幂等的），
简历则并发处理。
"""

from __future__ import annotations

import asyncio

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from multi_agent_resume_screener.agents.critic import critique
from multi_agent_resume_screener.agents.jd_parser import parse_jd
from multi_agent_resume_screener.pipeline.graph import build_pipeline
from multi_agent_resume_screener.state import (
    CandidateResult,
    JDRaw,
    PipelineConfig,
    PipelineState,
    ResumeRaw,
    ScreeningResult,
)


async def _run_one(
    pipeline,
    resume_raw: ResumeRaw,
    jd_raw: JDRaw,
    jd_structured,
    config: PipelineConfig,
) -> PipelineState:
    state = PipelineState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        jd_structured=jd_structured,  # 只解析一次，处处复用（parser 节点会跳过）
        config=config,
    )
    # pipeline.invoke 是阻塞调用；放到事件循环之外执行以实现并发。
    result = await asyncio.to_thread(pipeline.invoke, state)
    if isinstance(result, PipelineState):
        return result
    return PipelineState.model_validate(result)


async def screen(
    resumes: list[ResumeRaw],
    jd: JDRaw,
    config: PipelineConfig | None = None,
    llm: BaseChatModel | None = None,
    embedder: Embeddings | None = None,
) -> ScreeningResult:
    """针对一份职位描述筛选一份或多份简历。

    Args:
        resumes: 待筛选的简历（candidate 模式一份；recruiter 模式多份）。
        jd: 职位描述。
        config: 流水线配置（mode、critic_mode、weights 等）。
        llm: 可选的对话模型（用于测试的依赖注入）。
        embedder: 可选的嵌入客户端（用于测试的依赖注入）。

    Returns:
        一个 :class:`ScreeningResult`，候选人按分数降序排列。
    """
    config = config or PipelineConfig()
    jd_structured = parse_jd(jd, llm=llm)
    pipeline = build_pipeline(llm=llm, embedder=embedder)

    # 在招聘者 "fast" 模式下，先不带 critic 给所有人打分，之后只评估
    # 前 K 名。否则所有人都在流水线中内联运行 critic。
    fast = config.mode == "recruiter" and config.critic_mode == "fast"
    run_config = config.model_copy(update={"enable_critic": not fast})

    states = await asyncio.gather(
        *(_run_one(pipeline, r, jd, jd_structured, run_config) for r in resumes)
    )

    if fast:
        ranked_states = sorted(
            states,
            key=lambda s: s.final_score.score if s.final_score else 0.0,
            reverse=True,
        )
        for state in ranked_states[: config.critic_top_k]:
            state.critique = critique(
                state.resume_structured,
                state.jd_structured,
                state.match_result,
                state.final_score,
                config,
                hygiene=state.hygiene,
                llm=llm,
            )

    candidates = sorted(
        (CandidateResult.from_state(s) for s in states),
        key=lambda c: c.score,
        reverse=True,
    )
    return ScreeningResult(
        mode=config.mode,
        job_title=jd_structured.title,
        candidates=candidates,
    )
