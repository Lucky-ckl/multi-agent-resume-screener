"""针对若干 golden 查询样例，使用真实 embedder 打印未经过滤的原始
余弦相似度分数，用于校准 pipeline/retrieval.py 中的 `min_similarity`。

min_similarity=0.3 这个阈值当初是针对离线的粗糙词法假 embedder
（tests/test_retrieval.py 的 bag-of-words 哈希向量化器）调优的。已知
真实的稠密 embedding 模型（包括 Gemini）对*任意*两段自然语言给出
的基线相似度都远高于 bag-of-words 模型，因此该阈值对真实 API 可能
起不到任何区分作用。本脚本完全绕过 retrieve_evidence() 的阈值与
top-k 过滤，按降序打印每个候选 chunk 的原始分数，以便在确定新
数值之前看清真实分布。

用法：python scripts/inspect_similarity.py   （需要在 .env 中配置 GOOGLE_API_KEY）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_dataset import BACKEND_RESUME, DEVOPS_RESUME, FRONTEND_RESUME, JD_BACKEND, JD_DEVOPS, JD_ML  # noqa: E402

from dotenv import load_dotenv

from multi_agent_resume_screener.embeddings.client import get_embedder
from multi_agent_resume_screener.pipeline.retrieval import _cosine, _section_query, build_chunks
from multi_agent_resume_screener.state import Section, StructuredJD, StructuredResume


def _inspect(label: str, resume: StructuredResume, jd: StructuredJD, section: Section, embedder) -> None:
    chunks = [c for c in build_chunks(resume) if c.section == section]
    query = _section_query(section, jd)
    print(f"\n=== {label} ({section}) ===")
    print(f"query: {query!r}")
    if not query.strip() or not chunks:
        print("  (no query or no chunks for this section -- skipped)")
        return

    query_vec = embedder.embed_documents([query])[0]
    chunk_vecs = embedder.embed_documents([c.text for c in chunks])

    scored = sorted(
        ((c, _cosine(query_vec, v)) for c, v in zip(chunks, chunk_vecs)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for chunk, score in scored:
        title = chunk.title or "(skills/no title)"
        preview = chunk.text[:60].replace("\n", " ")
        print(f"  score={score:.4f}  [{title}]  {preview}...")


def main() -> int:
    load_dotenv()
    embedder = get_embedder()

    # 场景 1：正向查询，同一 section 内存在真实干扰项——
    # 确认相关条目的分数显著高于干扰项。
    _inspect("backend resume vs backend JD (has a decoy item)", BACKEND_RESUME, JD_BACKEND, "experience", embedder)
    _inspect("backend resume vs backend JD (has a decoy item)", BACKEND_RESUME, JD_BACKEND, "projects", embedder)
    _inspect("backend resume vs backend JD (true match, skills has no decoy)", BACKEND_RESUME, JD_BACKEND, "skills", embedder)
    _inspect("backend resume vs backend JD (true match, education has no decoy)", BACKEND_RESUME, JD_BACKEND, "education", embedder)

    # 场景 2：负对照，两个领域共享一个真实技能（Python）——
    # 评测中标出的"泄漏"可能只是合理的部分相关性，而非 bug。
    _inspect("backend resume vs ML JD (shares 'Python')", BACKEND_RESUME, JD_ML, "skills", embedder)
    _inspect("backend resume vs ML JD (shares 'Python')", BACKEND_RESUME, JD_ML, "experience", embedder)

    # 场景 3：完全没有任何共享技能的负对照——这是检验阈值对真实
    # 场景是否有区分度的最干净测试。
    _inspect("frontend resume vs DevOps JD (zero skill overlap)", FRONTEND_RESUME, JD_DEVOPS, "skills", embedder)
    _inspect("frontend resume vs DevOps JD (zero skill overlap)", FRONTEND_RESUME, JD_DEVOPS, "experience", embedder)

    # 场景 4：与场景 3 相同，但换一对不相关的组合，为"完全不相关"
    # 基线再提供一个数据点。
    _inspect("DevOps resume vs Backend JD (zero skill overlap)", DEVOPS_RESUME, JD_BACKEND, "skills", embedder)

    print(
        "\nDone. Compare the RELEVANT item's score against the decoy's score in "
        "case 1, and look at how high cases 3-4's scores are (should be clearly "
        "lower than case 1's relevant item if the embedding space discriminates "
        "unrelated domains). Paste this full output back for a threshold recommendation."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
