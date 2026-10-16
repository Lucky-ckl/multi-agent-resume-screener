"""证据检索（RAG）层的测试（离线，使用假 embedder）。"""

from __future__ import annotations

import math
import re
import zlib

import pytest

from multi_agent_resume_screener.embeddings.client import EmbeddingConfigError, get_embedder
from multi_agent_resume_screener.pipeline.retrieval import build_chunks, retrieve_evidence
from multi_agent_resume_screener.settings import Settings
from multi_agent_resume_screener.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    StructuredJD,
    StructuredResume,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 生产环境默认值（retrieval.py 的 _MIN_SIMILARITY，当前为 0.80）是
# 针对真实 Gemini embedding 分数校准的，对这个假 embedder 粗糙的
# 词法哈希毫无意义——后者产出的分数范围要低得多，且分布形态
# 也不相同。需要在“确实找到了匹配”与“什么都没找到”之间
# 作出区分的测试，会显式传入这个较低的阈值，
# 而不是默默沿用按生产环境调校的默认值。
_FAKE_MIN_SIMILARITY = 0.05


class _FakeEmbedder:
    """离线测试用的确定性词袋 embedder。

    生产环境的检索使用 Gemini 的 embedding API（见
    embeddings/client.py）；测试中用这个假 embedder 代替它，因此测试套件
    无需访问网络或 GOOGLE_API_KEY，同时其产出向量对关键词重叠的反映
    足够贴近，足以验证分块/排序逻辑。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 模拟真实 Gemini API 的行为：它会对空白文本返回 400
        # （EmbedContentRequest.content 包含空的 Part）——刻意如此严格，
        # 这样凡是发送空白文本的检索代码都会在这里显式报错，
        # 而不是只在生产环境中才暴露问题。
        for t in texts:
            if not t.strip():
                raise ValueError("embed_documents() received blank text")
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


def _resume_with_separable_facts() -> StructuredResume:
    """一份事实划分清晰、用于排序测试的合成简历。"""
    return StructuredResume(
        name="Jordan Candidate",
        skills=["python", "fastapi", "sql", "postgresql"],
        experience=[
            ExperienceItem(
                company="Acme Web Co",
                role="Backend Engineer",
                dates="2022-2024",
                bullets=[
                    "Built REST APIs in Python using FastAPI",
                    "Optimized PostgreSQL queries for a SQL-heavy reporting service",
                ],
            ),
        ],
        projects=[
            ProjectItem(
                name="Riverside Bridge Load Analysis",
                description="Structural load analysis for a pedestrian bridge, "
                "civil engineering coursework project.",
                tech=["AutoCAD", "structural analysis"],
            ),
        ],
        education=[
            EducationItem(degree="B.Tech Civil Engineering", institute="State University", year="2021"),
        ],
    )


# --------------------------------------------------------------------------- #
# 分块
# --------------------------------------------------------------------------- #
def test_build_chunks_one_per_item_plus_one_skills_chunk():
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)

    # 1 条经历 + 1 个项目 + 1 条教育 + 1 个技能块 = 4
    assert len(chunks) == 4
    sections = [c.section for c in chunks]
    assert sections.count("experience") == 1
    assert sections.count("projects") == 1
    assert sections.count("education") == 1
    assert sections.count("skills") == 1

    by_section = {c.section: c for c in chunks}
    assert by_section["experience"].title == "Backend Engineer at Acme Web Co"
    assert "FastAPI" in by_section["experience"].text
    assert by_section["projects"].title == "Riverside Bridge Load Analysis"
    assert by_section["education"].title == "State University"
    assert by_section["skills"].title is None
    assert "python" in by_section["skills"].text


def test_build_chunks_skips_fully_empty_items():
    resume = StructuredResume(
        experience=[ExperienceItem()],  # 所有字段均为空
        projects=[ProjectItem()],
        education=[EducationItem()],
    )
    chunks = build_chunks(resume)
    assert chunks == []


def test_build_chunks_no_skills_chunk_when_skills_empty():
    resume = StructuredResume(skills=[])
    chunks = build_chunks(resume)
    assert all(c.section != "skills" for c in chunks)


# --------------------------------------------------------------------------- #
# Embedder 确定性
# --------------------------------------------------------------------------- #
def test_fake_embedder_vectorize_is_deterministic():
    embedder = _FakeEmbedder()
    a = embedder.embed_documents(["Python FastAPI backend"])[0]
    b = embedder.embed_documents(["Python FastAPI backend"])[0]
    assert a == b


def test_get_embedder_raises_without_api_key():
    settings = Settings(google_api_key=None, llm_provider="groq")
    with pytest.raises(EmbeddingConfigError):
        get_embedder(settings=settings)


# --------------------------------------------------------------------------- #
# 检索排序
# --------------------------------------------------------------------------- #
def test_retrieval_ranks_relevant_chunk_highest():
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)
    jd = StructuredJD(
        title="Backend Engineer",
        required_skills=["python", "fastapi"],
        responsibilities=["Build Python backend services"],
    )

    evidence = retrieve_evidence(chunks, jd, embedder=_FakeEmbedder(), top_k=4, min_similarity=_FAKE_MIN_SIMILARITY)

    assert evidence["experience"], "expected the Python/FastAPI experience chunk to be retrieved"
    top = evidence["experience"][0]
    assert "FastAPI" in top.chunk.text
    assert top.chunk.title == "Backend Engineer at Acme Web Co"


def test_retrieval_returns_empty_when_no_relevant_evidence():
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)
    # 这份简历中没有任何机器学习相关内容。
    jd = StructuredJD(
        title="ML Engineer",
        required_skills=["machine learning", "tensorflow"],
        responsibilities=["Train and deploy machine learning models"],
    )

    evidence = retrieve_evidence(chunks, jd, embedder=_FakeEmbedder(), top_k=4)

    assert evidence["skills"] == []


def test_retrieval_skips_section_with_empty_jd_query_without_crashing():
    # 没有 qualifications 的 JD 会为教育板块生成空查询串。真实的 Gemini
    # embedding API 会以 400 拒绝空内容 Part——在 retrieve_evidence()
    # 学会跳过空查询板块之前，“简历含教育块、JD 无 qualifications”
    # 这一组合曾导致线上运行崩溃。
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)
    jd = StructuredJD(required_skills=["python"], responsibilities=["Python backend work"])
    assert jd.qualifications == []  # 触发空查询的条件

    evidence = retrieve_evidence(chunks, jd, embedder=_FakeEmbedder(), top_k=4)

    assert evidence["education"] == []  # 不崩溃，也不伪造证据


def test_retrieval_respects_top_k():
    chunks = build_chunks(
        StructuredResume(
            experience=[
                ExperienceItem(company=f"Company {i}", role="Engineer", bullets=[f"Did Python task {i}"])
                for i in range(6)
            ]
        )
    )
    jd = StructuredJD(required_skills=["python"], responsibilities=["Python engineering"])

    evidence = retrieve_evidence(
        chunks, jd, embedder=_FakeEmbedder(), top_k=2, min_similarity=_FAKE_MIN_SIMILARITY
    )

    # 恰好等于 top_k，而不是仅仅 "<= top_k"——6 个候选的匹配度都
    # 相近，因此这里真正验证了截断逻辑；若没有结果超过阈值，
    # 断言会空泛地通过。
    assert len(evidence["experience"]) == 2


def test_retrieval_no_chunks_returns_empty_sections():
    evidence = retrieve_evidence([], StructuredJD(required_skills=["python"]), embedder=_FakeEmbedder())
    assert all(v == [] for v in evidence.values())
