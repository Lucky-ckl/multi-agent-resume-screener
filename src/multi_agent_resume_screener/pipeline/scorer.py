"""确定性评分器：把 matcher 的子分数合成为最终分数。

这是有意设计成普通函数而非智能体的。保持最终分数的确定性意味着它可复现、
可单元测试、可调参（通过权重），而无需改动任何 LLM 提示词。给定相同的
``MatchResult`` 和权重，输出永远不变。

    final_score = Σ weightᵢ · sub_scoreᵢ      （板块 i）

由于 ``PipelineConfig`` 会把权重归一化到总和为 1.0，且每个子分数都在
[0, 1] 内，最终分数始终落在 [0, 1] 内。
"""

from __future__ import annotations

from multi_agent_resume_screener.state import (
    SECTIONS,
    FinalScore,
    MatchResult,
    PipelineConfig,
)


def score(match_result: MatchResult, config: PipelineConfig) -> FinalScore:
    """计算确定性的加权最终分数。

    Args:
        match_result: matcher 给出的各板块子分数。
        config: 提供板块权重（已归一化）的流水线配置。

    Returns:
        一个 :class:`FinalScore`，包含总分、各板块明细以及实际使用的权重
        （便于解释/审计）。
    """
    breakdown: dict[str, float] = {
        sub.section: sub.score
        for sub in match_result.sub_scores
        if sub.section in SECTIONS
    }
    # matcher 遗漏的任何板块按 0 计。
    for section in SECTIONS:
        breakdown.setdefault(section, 0.0)

    weights = config.weights
    total = sum(weights.get(section, 0.0) * breakdown[section] for section in SECTIONS)
    # 对微小的浮点漂移做截断，保证结果始终落在 [0, 1] 内。
    total = max(0.0, min(1.0, total))

    return FinalScore(
        score=total,
        weights_used=dict(weights),
        breakdown=breakdown,
    )
