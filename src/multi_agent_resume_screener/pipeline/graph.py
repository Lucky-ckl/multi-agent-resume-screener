"""用 LangGraph 把各智能体装配成一条筛选流水线。

流程（针对每份简历）::

    START
      → parse_resume        （智能体，已结构化则跳过）
      → parse_jd            （智能体，已结构化则跳过）
      → retrieve_evidence   （切分简历，并通过 Gemini 嵌入为每个板块
                              检索 top-k 证据；已计算则跳过，
                              否则为确定性执行）
      → match               （智能体；以检索到的证据为依据，重试时
                              使用 critic 的反馈）
      → score               （确定性）
      → hygiene             （确定性）
      → critic              （智能体；以确定性方式设置 needs_rescore）
      → [条件分支]
            ├─ prepare_retry → match     （一次自我纠错回路；重试时
                                           不重建检索）
            └─ END

唯一的条件边（critic 置信度低时回到 matcher，受 ``config.max_retries``
限制）正是它成为真正的 LangGraph 状态机、而非一条直线函数调用的原因。

每个节点通过返回扩展后的列表向 ``state.trace`` 追加记录，因此审计日志
无需自定义 channel reducer 就能不断累积。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from multi_agent_resume_screener.agents.critic import critique
from multi_agent_resume_screener.agents.jd_parser import parse_jd
from multi_agent_resume_screener.agents.matcher import match
from multi_agent_resume_screener.agents.parser import parse_resume
from multi_agent_resume_screener.pipeline.hygiene import check_hygiene
from multi_agent_resume_screener.pipeline.retrieval import build_chunks, retrieve_evidence
from multi_agent_resume_screener.pipeline.scorer import score
from multi_agent_resume_screener.state import (
    JDRaw,
    PipelineConfig,
    PipelineState,
    ResumeRaw,
    TraceEntry,
)


def build_pipeline(llm: BaseChatModel | None = None, embedder: Embeddings | None = None):
    """构建并编译筛选流水线图。

    Args:
        llm: 注入每个智能体的可选对话模型（测试用它离线运行图）。为
            ``None`` 时，每个智能体自行解析所配置的提供商。
        embedder: 注入检索节点的可选嵌入客户端（测试用它离线运行检索）。
            为 ``None`` 时，自行解析所配置的 Gemini 嵌入器。

    Returns:
        编译后的 LangGraph 可运行对象。传入 :class:`PipelineState`
        （或等价的字典）调用它，将返回最终状态。
    """

    def _trace(state: PipelineState, agent: str, note: str = "") -> list[TraceEntry]:
        return state.trace + [TraceEntry(agent=agent, note=note)]

    def parse_resume_node(state: PipelineState) -> dict:
        if state.resume_structured is not None:
            return {}
        structured = parse_resume(state.resume_raw, llm=llm)
        return {
            "resume_structured": structured,
            "trace": _trace(state, "parser", f"{len(structured.skills)} skills"),
        }

    def parse_jd_node(state: PipelineState) -> dict:
        if state.jd_structured is not None:
            return {}
        structured = parse_jd(state.jd_raw, llm=llm)
        return {
            "jd_structured": structured,
            "trace": _trace(state, "jd_parser", structured.title or ""),
        }

    def retrieve_evidence_node(state: PipelineState) -> dict:
        if state.resume_chunks:
            return {}
        chunks = build_chunks(state.resume_structured)
        evidence = retrieve_evidence(
            chunks, state.jd_structured, embedder=embedder, top_k=state.config.retrieval_top_k
        )
        hits = sum(len(v) for v in evidence.values())
        return {
            "resume_chunks": chunks,
            "retrieved_evidence": evidence,
            "trace": _trace(state, "retriever", f"{len(chunks)} chunks, {hits} retrieved"),
        }

    def match_node(state: PipelineState) -> dict:
        # 重试时，把 critic 的建议反馈给 matcher。
        feedback = None
        if state.retry_count > 0 and state.critique is not None:
            feedback = state.critique.suggestions
        result = match(
            state.resume_structured,
            state.jd_structured,
            llm=llm,
            feedback=feedback,
            retrieved_evidence=state.retrieved_evidence,
        )
        note = f"retry={state.retry_count}" if feedback else "initial"
        return {"match_result": result, "trace": _trace(state, "matcher", note)}

    def score_node(state: PipelineState) -> dict:
        final = score(state.match_result, state.config)
        return {
            "final_score": final,
            "trace": _trace(state, "scorer", f"score={final.score:.2f}"),
        }

    def hygiene_node(state: PipelineState) -> dict:
        report = check_hygiene(state.resume_structured, state.resume_raw.text)
        return {
            "hygiene": report,
            "trace": _trace(state, "hygiene", f"{len(report.issues)} issues"),
        }

    def critic_node(state: PipelineState) -> dict:
        result = critique(
            state.resume_structured,
            state.jd_structured,
            state.match_result,
            state.final_score,
            state.config,
            hygiene=state.hygiene,
            llm=llm,
        )
        note = f"{result.verdict}, rescore={result.needs_rescore}"
        return {"critique": result, "trace": _trace(state, "critic", note)}

    def prepare_retry_node(state: PipelineState) -> dict:
        return {
            "retry_count": state.retry_count + 1,
            "trace": _trace(state, "self_correction", "looping back to matcher"),
        }

    def route_after_critic(state: PipelineState) -> str:
        if (
            state.critique is not None
            and state.critique.needs_rescore
            and state.can_retry()
        ):
            return "retry"
        return "end"

    def route_after_hygiene(state: PipelineState) -> str:
        # 快速模式（enable_critic=False）在打分 + 体检后即结束；
        # critic 稍后只对排序前 K 名的候选人单独运行。
        return "critic" if state.config.enable_critic else "end"

    graph = StateGraph(PipelineState)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("match", match_node)
    graph.add_node("score", score_node)
    graph.add_node("check_hygiene", hygiene_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_retry", prepare_retry_node)

    graph.add_edge(START, "parse_resume")
    graph.add_edge("parse_resume", "parse_jd")
    graph.add_edge("parse_jd", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "match")
    graph.add_edge("match", "score")
    graph.add_edge("score", "check_hygiene")
    graph.add_conditional_edges(
        "check_hygiene",
        route_after_hygiene,
        {"critic": "critic", "end": END},
    )
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "prepare_retry", "end": END},
    )
    graph.add_edge("prepare_retry", "match")

    return graph.compile()


def run_pipeline(
    resume_raw: ResumeRaw,
    jd_raw: JDRaw,
    config: PipelineConfig | None = None,
    llm: BaseChatModel | None = None,
    embedder: Embeddings | None = None,
) -> PipelineState:
    """让一份简历跑完整条流水线并返回最终状态。

    Args:
        resume_raw: 简历文件名 + 提取出的文本。
        jd_raw: 原始职位描述文本。
        config: 可选的流水线配置（mode、weights、retries 等）。
        llm: 可选的对话模型（用于测试的依赖注入）。
        embedder: 可选的嵌入客户端（用于测试的依赖注入）。

    Returns:
        最终的 :class:`PipelineState`，包含结构化数据、分数、体检报告、
        评估结果和累积的审计轨迹。
    """
    pipeline = build_pipeline(llm=llm, embedder=embedder)
    initial = PipelineState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        config=config or PipelineConfig(),
    )
    result = pipeline.invoke(initial)
    # LangGraph 可能返回类字典对象；归一化回 PipelineState。
    if isinstance(result, PipelineState):
        return result
    return PipelineState.model_validate(result)
