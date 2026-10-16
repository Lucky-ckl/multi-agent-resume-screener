"""multi-agent-resume-screener：多智能体简历筛选流水线。

各智能体（parser、jd_parser、matcher、critic）围绕共享的
:class:`multi_agent_resume_screener.state.PipelineState` 由 LangGraph 编排。
评分是一个确定性函数，因此最终分数始终可复现、可审计。
"""

__version__ = "0.1.0"
