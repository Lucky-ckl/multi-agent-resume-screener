"""LangGraph 流水线编排的测试（离线，使用路由式假 LLM）。"""

from __future__ import annotations

import math
import re
import zlib

from multi_agent_resume_screener.agents.critic import _CriticLLMOutput
from multi_agent_resume_screener.pipeline.graph import build_pipeline, run_pipeline
from multi_agent_resume_screener.state import (
    CandidateResult,
    JDRaw,
    MatchResult,
    PipelineConfig,
    ResumeRaw,
    StructuredJD,
    StructuredResume,
    SubScore,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _FakeEmbedder:
    """离线测试用的确定性词袋 embedder（与上面的假 LLM 采用同一
    模式；针对该假 embedder 的检索专项测试见 tests/test_retrieval.py）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    @staticmethod
    def _vectorize(text: str, dim: int = 64) -> list[float]:
        vec = [0.0] * dim
        for tok in _TOKEN_RE.findall(text.lower()):
            vec[zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class _RoutingRunnable:
    def __init__(self, schema, results, counts):
        self._schema = schema
        self._results = results
        self._counts = counts

    def invoke(self, messages):
        self._counts[self._schema.__name__] = (
            self._counts.get(self._schema.__name__, 0) + 1
        )
        return self._results[self._schema.__name__]


class _RoutingFakeLLM:
    """根据请求的 schema 名称分发 with_structured_output()。"""

    def __init__(self, results: dict):
        self.results = results
        self.counts: dict[str, int] = {}

    def with_structured_output(self, schema, **kwargs):
        return _RoutingRunnable(schema, self.results, self.counts)


def _results(confidence: float) -> dict:
    return {
        "StructuredResume": StructuredResume(name="Jane", skills=["python", "go"]),
        "StructuredJD": StructuredJD(title="Backend", required_skills=["python"]),
        "MatchResult": MatchResult(
            sub_scores=[SubScore(section="skills", score=0.8)],
            overall_evidence_quality=0.9,
        ),
        "_CriticLLMOutput": _CriticLLMOutput(
            gaps=["Missing Kafka"],
            suggestions=["Add a Kafka project"],
            verdict="moderate_fit",
            confidence_in_scoring=confidence,
        ),
    }


def _inputs():
    return (
        ResumeRaw(filename="jane.pdf", text="Jane\nSkills: python, go"),
        JDRaw(text="Backend engineer, Python required"),
    )


def test_happy_path_no_retry():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.95))  # 置信度高 -> 不进入循环

    state = run_pipeline(resume_raw, jd_raw, PipelineConfig(), llm=llm, embedder=_FakeEmbedder())

    assert state.resume_structured.name == "Jane"
    assert state.jd_structured.title == "Backend"
    assert state.match_result is not None
    assert state.final_score is not None
    assert state.hygiene is not None
    assert state.critique.verdict == "moderate_fit"
    assert state.retry_count == 0
    # matcher 和 critic 各恰好运行了一次。
    assert llm.counts["MatchResult"] == 1
    assert llm.counts["_CriticLLMOutput"] == 1
    # Trace 按顺序记录了每个阶段。
    agents = [t.agent for t in state.trace]
    assert agents == ["parser", "jd_parser", "retriever", "matcher", "scorer", "hygiene", "critic"]


def test_self_correction_loop_runs_once():
    resume_raw, jd_raw = _inputs()
    # 置信度低 -> needs_rescore -> 恰好回环到 matcher 一次。
    llm = _RoutingFakeLLM(_results(confidence=0.2))

    state = run_pipeline(
        resume_raw, jd_raw, PipelineConfig(max_retries=1, confidence_threshold=0.6),
        llm=llm, embedder=_FakeEmbedder(),
    )

    assert state.retry_count == 1
    # matcher 运行了两次（初始 + 一次重试）；critic 同样运行了两次。
    assert llm.counts["MatchResult"] == 2
    assert llm.counts["_CriticLLMOutput"] == 2
    # 解析器只运行了一次（第二遍幂等跳过）。
    assert llm.counts["StructuredResume"] == 1
    assert llm.counts["StructuredJD"] == 1
    assert any(t.agent == "self_correction" for t in state.trace)


def test_retrieval_runs_once_and_survives_retry():
    resume_raw, jd_raw = _inputs()
    # 置信度低 -> 重试一次，但检索不能被重新计算。
    llm = _RoutingFakeLLM(_results(confidence=0.2))

    state = run_pipeline(
        resume_raw, jd_raw, PipelineConfig(max_retries=1, confidence_threshold=0.6),
        llm=llm, embedder=_FakeEmbedder(),
    )

    assert llm.counts["MatchResult"] == 2  # matcher 重试了
    assert state.resume_chunks  # 检索已填充
    retriever_entries = [t for t in state.trace if t.agent == "retriever"]
    assert len(retriever_entries) == 1  # 但检索只运行了一次


def test_retry_disabled_when_max_retries_zero():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.1))  # 本应触发重打分

    state = run_pipeline(
        resume_raw, jd_raw, PipelineConfig(max_retries=0), llm=llm, embedder=_FakeEmbedder(),
    )

    assert state.retry_count == 0
    assert llm.counts["MatchResult"] == 1  # 没有循环


def test_pipeline_result_converts_to_candidate_result():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.95))
    state = run_pipeline(resume_raw, jd_raw, PipelineConfig(), llm=llm, embedder=_FakeEmbedder())

    result = CandidateResult.from_state(state)
    assert result.filename == "jane.pdf"
    assert result.candidate_name == "Jane"
    assert result.verdict == "moderate_fit"
    assert 0.0 <= result.score <= 1.0
    assert result.hygiene_score is not None


def test_prebuilt_pipeline_is_reusable():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.95))
    pipeline = build_pipeline(llm=llm, embedder=_FakeEmbedder())

    from multi_agent_resume_screener.state import PipelineState

    out = pipeline.invoke(
        PipelineState(resume_raw=resume_raw, jd_raw=jd_raw, config=PipelineConfig())
    )
    state = out if isinstance(out, PipelineState) else PipelineState.model_validate(out)
    assert state.critique is not None
