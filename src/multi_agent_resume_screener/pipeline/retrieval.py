"""证据检索：切分简历、嵌入向量、按板块检索 top-k。

这是 RAG 层，让 matcher 的分板块评分建立在具体的简历摘录之上，而不只依赖
整份简历。除嵌入调用本身外它完全确定性——嵌入调用委托给注入的
``Embeddings`` 客户端（生产环境用 Gemini；测试用假实现——见
:mod:`multi_agent_resume_screener.embeddings.client`）。

分块粒度是每条简历条目一个单元（每条经历/项目/教育条目一个，整份技能
列表一个），因此每个分块都是一段有意义、自包含的证据，并带有真实的
元数据（公司/职位、项目名、院校）。检索会根据 ``StructuredJD`` 的相关
字段为每个板块构建一个查询，嵌入后按余弦相似度对该板块自己的分块排序
——低于相关性下限的结果一律丢弃，这样没有任何真正相关证据的板块会返回
空列表，而不是被无关的 top-k 结果凑数。
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

from multi_agent_resume_screener.embeddings.client import get_embedder
from multi_agent_resume_screener.state import (
    SECTIONS,
    ResumeChunk,
    RetrievedChunk,
    Section,
    StructuredJD,
    StructuredResume,
)

# 低于该余弦相似度的分块不视为相关证据。
#
# 该阈值是针对真实 Gemini 嵌入器实证校准的（见
# scripts/inspect_similarity.py），并非拍脑袋：同领域的简历/JD 匹配在
# skills/experience/projects 上的得分为 0.850-0.914，而所有跨领域（技能
# 部分重叠或零重叠）配对的得分最高只有 0.789 —— 稠密嵌入会把简短的
# 简历/JD 文本压缩到一个比词法模型窄得多、相似度更高的区间，因此针对
# 其中一类调优的阈值对另一类毫无意义。0.80 恰好落在观测到的空隙中。
# 这只是一个小规模的经验样本（少量领域配对）；如果生产使用中发现
# matcher 经常在理应匹配的内容上得到"未检索到证据"，应重新审视该阈值。
_MIN_SIMILARITY = 0.80


def build_chunks(resume: StructuredResume) -> list[ResumeChunk]:
    """把结构化简历切分为可检索、语义上有意义的分块。

    每条经历/项目/教育条目一个分块（完全为空的条目会被跳过），
    整份技能列表再额外一个分块。
    """
    chunks: list[ResumeChunk] = []

    for exp in resume.experience:
        if not (exp.company or exp.role or exp.bullets):
            continue
        title = " at ".join(p for p in (exp.role, exp.company) if p) or None
        text = "\n".join(filter(None, [title, exp.dates, *exp.bullets]))
        chunks.append(ResumeChunk(section="experience", title=title, text=text))

    for proj in resume.projects:
        if not (proj.name or proj.description or proj.tech):
            continue
        text = "\n".join(filter(None, [proj.description, ", ".join(proj.tech)]))
        chunks.append(ResumeChunk(section="projects", title=proj.name, text=text))

    for edu in resume.education:
        if not (edu.degree or edu.institute or edu.year or edu.gpa):
            continue
        text = " ".join(filter(None, [edu.degree, edu.institute, edu.year, edu.gpa]))
        chunks.append(ResumeChunk(section="education", title=edu.institute, text=text))

    if resume.skills:
        chunks.append(
            ResumeChunk(section="skills", title=None, text=", ".join(resume.skills))
        )

    return chunks


def _section_query(section: Section, jd: StructuredJD) -> str:
    """根据结构化 JD 为一个板块构建检索查询文本。"""
    if section == "skills":
        return " ".join(jd.required_skills + jd.nice_to_have_skills)
    if section == "experience":
        parts = list(jd.responsibilities)
        if jd.title:
            parts.append(jd.title)
        if jd.min_experience_years:
            parts.append(f"{jd.min_experience_years} years of experience")
        return " ".join(parts)
    if section == "projects":
        return " ".join(jd.responsibilities + jd.required_skills)
    return " ".join(jd.qualifications)  # education 板块


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_evidence(
    chunks: list[ResumeChunk],
    jd: StructuredJD,
    embedder: Embeddings | None = None,
    top_k: int = 4,
    min_similarity: float = _MIN_SIMILARITY,
) -> dict[str, list[RetrievedChunk]]:
    """按板块检索最相关的 top-k 简历分块。

    Args:
        chunks: 简历的分块（见 :func:`build_chunks`）。
        jd: 结构化的职位描述。
        embedder: 可选的嵌入客户端（用于测试的依赖注入）。缺省回退到
            配置的 Gemini 嵌入器。
        top_k: 每个板块最多返回的分块数。
        min_similarity: 得分低于该值的分块会被丢弃，而不是用无关证据
            凑数。

    Returns:
        :data:`~multi_agent_resume_screener.state.SECTIONS` 中每个板块一项，
        每项是按分数降序排列的 :class:`RetrievedChunk` 列表
        （若该板块没有分块达到相关性下限则可能为空）。
    """
    embedder = embedder or get_embedder()

    by_section: dict[str, list[ResumeChunk]] = {section: [] for section in SECTIONS}
    for chunk in chunks:
        by_section[chunk.section].append(chunk)

    sections_with_chunks = [s for s in SECTIONS if by_section[s]]
    result: dict[str, list[RetrievedChunk]] = {section: [] for section in SECTIONS}
    if not sections_with_chunks:
        return result

    # 两类嵌入调用都批量执行（所有分块文本一次、所有板块查询一次），
    # 而不是每个分块/查询各调一次，以尽量减少 API 用量。
    # 空白文本会从两个批次中剔除：Gemini 的嵌入 API 会直接拒绝空的
    # content Part（400 错误），而且空的 JD 查询本来也没有可检索的信号，
    # 因此这些板块会正确地返回空证据，而不是让整个请求崩溃。
    all_chunks = [c for s in sections_with_chunks for c in by_section[s] if c.text.strip()]
    if not all_chunks:
        return result
    chunk_vectors = embedder.embed_documents([c.text for c in all_chunks])
    vectors_by_id = dict(zip((id(c) for c in all_chunks), chunk_vectors))
    chunked_sections = {c.section for c in all_chunks}

    queryable_sections = [
        s for s in sections_with_chunks if s in chunked_sections and _section_query(s, jd).strip()
    ]
    if not queryable_sections:
        return result

    queries = [_section_query(s, jd) for s in queryable_sections]
    query_vectors = embedder.embed_documents(queries)

    for section, query_vector in zip(queryable_sections, query_vectors):
        scored = sorted(
            (
                RetrievedChunk(chunk=c, score=_cosine(query_vector, vectors_by_id[id(c)]))
                for c in by_section[section]
                if id(c) in vectors_by_id
            ),
            key=lambda rc: rc.score,
            reverse=True,
        )
        result[section] = [rc for rc in scored[:top_k] if rc.score >= min_similarity]

    return result
