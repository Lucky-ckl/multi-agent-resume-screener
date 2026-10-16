"""在 golden 数据集上用 Recall@K 评估证据检索（RAG）层。

用法（在项目根目录、已激活虚拟环境并在 .env 中设置 GOOGLE_API_KEY 时）：

    python scripts/eval_retrieval.py                # 真实 Gemini embeddings
    python scripts/eval_retrieval.py --top-k 3
    python scripts/eval_retrieval.py --fake          # 试运行，不调用 API

默认调用真实的 Gemini embeddings API（与生产流水线所用相同），因为
值得写进简历的指标必须针对真实系统测量，而不是测试套件为封闭 CI
使用的离线假 embedder。--fake 仅用于在不消耗 API 配额的情况下对
评测脚本本身做健全性检查。

这里的 Recall@K =（出现在 top-K 检索结果中的 gold 相关 chunk 数）/
（该 section 存在的 gold 相关 chunk 数），并在所有存在至少一个相关
chunk 的（简历，JD，section）查询上取平均。相关 chunk 数为零的查询
（负对照）单独报告，作为"不捏造相关性"检查，因为它们的 recall
无定义（0/0）。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_dataset import GOLDEN_QUERIES, NEGATIVE_CONTROLS, GoldenQuery  # noqa: E402

from dotenv import load_dotenv

from multi_agent_resume_screener.embeddings.client import EmbeddingConfigError, get_embedder
from multi_agent_resume_screener.pipeline.retrieval import build_chunks, retrieve_evidence


def _group_by_resume_jd(queries: list[GoldenQuery]) -> list[dict]:
    """按（简历，JD）对对共享同一组合的 golden 查询分组，使检索每组
    只运行一次而非每个 section 一次——减少 embedding API 调用次数。"""
    groups: dict[tuple[int, int], dict] = {}
    order: list[tuple[int, int]] = []
    for q in queries:
        key = (id(q.resume), id(q.jd))
        if key not in groups:
            groups[key] = {"resume": q.resume, "jd": q.jd, "queries": []}
            order.append(key)
        groups[key]["queries"].append(q)
    return [groups[k] for k in order]


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
_MAX_RETRIES = 5
_DEFAULT_RETRY_DELAY = 40.0  # 秒；稳妥地大于免费档的每分钟窗口
_PACING_DELAY = 1.5  # （简历，JD）组之间的间隔秒数，从源头避免触发限流


def _retrieve_with_retry(chunks, jd, embedder, top_k, min_similarity):
    """调用 retrieve_evidence()，遇到免费档限流（429）时重试。

    评测脚本会背靠背地跑许多（简历，JD）组，即使单次真实筛选请求
    （1-2 次调用）永远不会触发，也可能用尽免费档每分钟的 embedding
    配额。若 API 错误信息中带建议延迟则按其重试，否则使用固定的
    回退延迟。

    ``min_similarity=None`` 表示"使用 retrieve_evidence() 自身的生产
    默认值"（让 retrieval.py 保持唯一事实来源，而不是在这里重复一份）
    ——用于真实 Gemini 运行。--fake 运行则传入显式的低阈值，因为假
    embedder 的分数量级与校准生产默认值所依据的真实分数量级不可比
    （见 retrieval.py 的 _MIN_SIMILARITY 注释）。
    """
    kwargs = {"top_k": top_k}
    if min_similarity is not None:
        kwargs["min_similarity"] = min_similarity
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return retrieve_evidence(chunks, jd, embedder=embedder, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 仅重试限流错误，其余原样抛出
            message = str(exc)
            is_rate_limit = "429" in message or "quota" in message.lower()
            if not is_rate_limit or attempt == _MAX_RETRIES:
                raise
            match = _RETRY_DELAY_RE.search(message)
            delay = float(match.group(1)) + 5 if match else _DEFAULT_RETRY_DELAY
            print(f"  [rate limited, attempt {attempt}/{_MAX_RETRIES}] waiting {delay:.0f}s before retry...")
            time.sleep(delay)


class _FakeEmbedder:
    """用于 --fake 试运行的确定性 bag-of-words embedder（与离线测试
    套件中的实现一致；从不用于对外发布的指标）。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import math
        import re
        import zlib

        # 模拟真实 Gemini API 的行为：空文本会被 400 拒绝——
        # 故意从严，让 --fake 试运行也能捕获空查询的 bug。
        for t in texts:
            if not t.strip():
                raise ValueError("embed_documents() received blank text")

        def vectorize(text: str, dim: int = 64) -> list[float]:
            vec = [0.0] * dim
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            return [v / norm for v in vec] if norm else vec

        return [vectorize(t) for t in texts]


_FAKE_MIN_SIMILARITY = 0.05  # 假 embedder 的分数量级与真实 Gemini 不可比


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=4, help="Retrieval top-k (default: 4, matches PipelineConfig default)")
    parser.add_argument("--fake", action="store_true", help="Use a deterministic fake embedder instead of the real Gemini API")
    parser.add_argument(
        "--min-similarity", type=float, default=None,
        help="Override the relevance floor. Defaults to retrieval.py's production value for real "
             f"runs, or {_FAKE_MIN_SIMILARITY} for --fake (the two embedders' score scales aren't comparable).",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.fake:
        embedder = _FakeEmbedder()
        min_similarity = args.min_similarity if args.min_similarity is not None else _FAKE_MIN_SIMILARITY
        print("Using FAKE embedder (dry run) -- these numbers are NOT for publishing.\n")
    else:
        try:
            embedder = get_embedder()
        except EmbeddingConfigError as exc:
            print(f"[CONFIG ERROR] {exc}")
            return 2
        min_similarity = args.min_similarity  # None -> 使用 retrieve_evidence() 自身的生产默认值
        print("Using the real Gemini embedder configured in your environment.\n")

    all_queries = GOLDEN_QUERIES + NEGATIVE_CONTROLS
    groups = _group_by_resume_jd(all_queries)

    per_query_recall: list[tuple[str, int, int]] = []  # id、命中数、相关总数
    negative_control_leaks: list[tuple[str, int]] = []  # id、检索到的条数（应为 0）

    print(f"{'query':<38} {'section':<12} {'result':<18}")
    print("-" * 70)

    for i, group in enumerate(groups):
        if i > 0 and not args.fake:
            time.sleep(_PACING_DELAY)  # 拉开真实 API 调用的间隔，避免触发配额限制
        chunks = build_chunks(group["resume"])
        try:
            evidence = _retrieve_with_retry(chunks, group["jd"], embedder, args.top_k, min_similarity)
        except Exception as exc:  # noqa: BLE001 - 直接抛出任何提供商/网络错误
            print(f"[REQUEST FAILED] {type(exc).__name__}: {exc}")
            return 1

        for q in group["queries"]:
            section_chunks = [c for c in chunks if c.section == q.section]
            relevant_chunks = [c for c in section_chunks if q.relevant(c.text)]
            retrieved_texts = {rc.chunk.text for rc in evidence.get(q.section, [])}

            if relevant_chunks:
                hits = sum(1 for c in relevant_chunks if c.text in retrieved_texts)
                per_query_recall.append((q.id, hits, len(relevant_chunks)))
                print(f"{q.id:<38} {q.section:<12} recall={hits}/{len(relevant_chunks)}")
            else:
                leaked = len(retrieved_texts)
                negative_control_leaks.append((q.id, leaked))
                status = "OK (nothing retrieved)" if leaked == 0 else f"LEAK ({leaked} retrieved!)"
                print(f"{q.id:<38} {q.section:<12} {status}")

    print("-" * 70)

    macro_recall = sum(h / t for _, h, t in per_query_recall) / len(per_query_recall)
    micro_hits = sum(h for _, h, _ in per_query_recall)
    micro_total = sum(t for _, _, t in per_query_recall)
    micro_recall = micro_hits / micro_total

    print(f"\nRecall@{args.top_k} (macro-avg over {len(per_query_recall)} queries): {macro_recall:.3f}")
    print(f"Recall@{args.top_k} (micro-avg, {micro_hits}/{micro_total} relevant chunks retrieved): {micro_recall:.3f}")

    leaks = sum(1 for _, n in negative_control_leaks if n > 0)
    print(f"\nNegative-control check ({len(negative_control_leaks)} unrelated-JD queries): "
          f"{len(negative_control_leaks) - leaks}/{len(negative_control_leaks)} correctly retrieved nothing")

    print(
        f"\nSuggested line: \"Built and evaluated a RAG evidence-retrieval layer "
        f"against a {len(per_query_recall)}-query golden dataset, achieving "
        f"Recall@{args.top_k} = {macro_recall:.2f}.\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
