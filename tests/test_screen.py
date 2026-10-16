"""筛选编排的测试（离线，使用路由式假 LLM）。"""

from __future__ import annotations

import asyncio
import math
import re
import zlib

from multi_agent_resume_screener.agents.critic import _CriticLLMOutput
from multi_agent_resume_screener.pipeline.screen import screen
from multi_agent_resume_screener.state import (
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
    """离线测试用的确定性词袋 embedder（针对该假 embedder 的
    检索专项测试见 tests/test_retrieval.py）。"""

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
        name = self._schema.__name__
        self._counts[name] = self._counts.get(name, 0) + 1
        result = self._results[name]
        return result(self._counts) if callable(result) else result


class _RoutingFakeLLM:
    def __init__(self, results: dict):
        self.results = results
        self.counts: dict[str, int] = {}

    def with_structured_output(self, schema, **kwargs):
        return _RoutingRunnable(schema, self.results, self.counts)


def _make_llm(score_for_resume) -> _RoutingFakeLLM:
    # MatchResult 的分数随调用次数变化，使各候选人分数各不相同。
    def match_result(counts):
        n = counts["MatchResult"]
        return MatchResult(
            sub_scores=[SubScore(section="skills", score=score_for_resume(n))],
            overall_evidence_quality=0.9,
        )

    return _RoutingFakeLLM(
        {
            "StructuredResume": StructuredResume(name="Cand", skills=["python"]),
            "StructuredJD": StructuredJD(title="Backend", required_skills=["python"]),
            "MatchResult": match_result,
            "_CriticLLMOutput": _CriticLLMOutput(
                gaps=[], suggestions=["s"], verdict="moderate_fit",
                confidence_in_scoring=0.95,
            ),
        }
    )


def _resumes(n: int) -> list[ResumeRaw]:
    return [ResumeRaw(filename=f"r{i}.pdf", text=f"resume {i}") for i in range(n)]


def test_candidate_mode_returns_single_ranked_result():
    llm = _make_llm(lambda n: 0.8)
    result = asyncio.run(
        screen(
            _resumes(1), JDRaw(text="jd"), PipelineConfig(mode="candidate"),
            llm=llm, embedder=_FakeEmbedder(),
        )
    )
    assert result.mode == "candidate"
    assert result.job_title == "Backend"
    assert len(result.candidates) == 1
    assert result.candidates[0].suggestions  # critic 已运行


def test_recruiter_full_ranks_all_and_critiques_all():
    # 按调用顺序给出不同分数，以便观察排序效果。
    scores = {1: 0.2, 2: 0.9, 3: 0.5}
    llm = _make_llm(lambda n: scores[n])
    result = asyncio.run(
        screen(
            _resumes(3), JDRaw(text="jd"),
            PipelineConfig(mode="recruiter", critic_mode="full"), llm=llm,
            embedder=_FakeEmbedder(),
        )
    )
    assert len(result.candidates) == 3
    # 按分数降序排列。
    out_scores = [c.score for c in result.candidates]
    assert out_scores == sorted(out_scores, reverse=True)
    # full 模式下每位候选人都被点评。
    assert all(c.verdict == "moderate_fit" for c in result.candidates)


def test_recruiter_fast_critiques_only_top_k():
    scores = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7}
    llm = _make_llm(lambda n: scores[n])
    result = asyncio.run(
        screen(
            _resumes(4), JDRaw(text="jd"),
            PipelineConfig(mode="recruiter", critic_mode="fast", critic_top_k=2),
            llm=llm, embedder=_FakeEmbedder(),
        )
    )
    assert len(result.candidates) == 4
    # critic 恰好只对前 2 名运行（他们有建议）。
    critiqued = [c for c in result.candidates if c.suggestions]
    assert len(critiqued) == 2
    # 分数最高的两名就是被点评的候选人。
    assert result.candidates[0].suggestions
    assert result.candidates[1].suggestions
    assert not result.candidates[2].suggestions
    assert llm.counts["_CriticLLMOutput"] == 2


def test_jd_parsed_once_across_resumes():
    llm = _make_llm(lambda n: 0.5)
    asyncio.run(
        screen(
            _resumes(3), JDRaw(text="jd"),
            PipelineConfig(mode="recruiter", critic_mode="full"), llm=llm,
            embedder=_FakeEmbedder(),
        )
    )
    # 尽管筛选了 3 份简历，JD 只被解析了一次。
    assert llm.counts["StructuredJD"] == 1
