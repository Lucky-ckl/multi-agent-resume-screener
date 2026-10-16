"""暴露筛选流水线的 FastAPI 应用。

端点：

- ``GET  /health`` — 存活检查。
- ``POST /screen`` — 上传一份或多份简历 PDF 以及职位描述，返回带排名、
  可解释的筛选结果。

角色由 ``mode`` 字段选择：``candidate``（简历反馈）或 ``recruiter``（排名
筛选）。
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from multi_agent_resume_screener import __version__
from multi_agent_resume_screener.embeddings.client import EmbeddingConfigError
from multi_agent_resume_screener.llm.client import LLMConfigError
from multi_agent_resume_screener.pdf import PDFExtractionError, extract_text_from_pdf
from multi_agent_resume_screener.pipeline.screen import screen
from multi_agent_resume_screener.settings import get_settings
from multi_agent_resume_screener.state import (
    JDRaw,
    PipelineConfig,
    ResumeRaw,
    ScreeningResult,
)
from multi_agent_resume_screener.storage.runs import RunStore

load_dotenv()

app = FastAPI(
    title="multi-agent-resume-screener",
    version=__version__,
    summary="Multi-agent resume screening (LangGraph + Gemini).",
)

# 提供随包打包的静态前端（HTML/CSS/JS）。
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def get_llm() -> BaseChatModel | None:
    """LLM 依赖。返回 ``None``，让各智能体使用已配置的提供方。

    测试通过 ``app.dependency_overrides`` 覆盖此项，注入假模型。
    """
    return None


def get_embedder() -> Embeddings | None:
    """Embeddings 依赖。返回 ``None``，让检索使用已配置的 Gemini embedder。

    测试通过 ``app.dependency_overrides`` 覆盖此项，注入假 embedder。
    """
    return None


@lru_cache(maxsize=1)
def _default_store() -> RunStore:
    return RunStore(get_settings().db_path)


def get_store() -> RunStore:
    """运行记录存储依赖。测试中会用临时文件存储覆盖此项。"""
    return _default_store()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    # 提供单页前端。交互式 API 文档仍位于 /docs。
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/screen", response_model=ScreeningResult)
async def screen_endpoint(
    jd: str = Form(..., description="Job description text."),
    mode: str = Form("recruiter", description="'candidate' or 'recruiter'."),
    critic_mode: str = Form("full", description="'fast' or 'full'."),
    critic_top_k: int = Form(5, ge=1),
    resumes: list[UploadFile] = File(..., description="Resume PDF file(s)."),
    llm: BaseChatModel | None = Depends(get_llm),
    embedder: Embeddings | None = Depends(get_embedder),
    store: RunStore = Depends(get_store),
) -> ScreeningResult:
    if not jd.strip():
        raise HTTPException(status_code=422, detail="Job description is empty.")
    if not resumes:
        raise HTTPException(status_code=422, detail="At least one resume is required.")

    # 先构建 config，以便清晰地报告无效的 mode/critic_mode。
    try:
        config = PipelineConfig(
            mode=mode,  # type: ignore[arg-type]
            critic_mode=critic_mode,  # type: ignore[arg-type]
            critic_top_k=critic_top_k,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    resume_inputs: list[ResumeRaw] = []
    for upload in resumes:
        raw = await upload.read()
        try:
            text = extract_text_from_pdf(raw)
        except PDFExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{upload.filename}': {exc}",
            ) from exc
        resume_inputs.append(
            ResumeRaw(filename=upload.filename or "resume.pdf", text=text)
        )

    try:
        result = await screen(resume_inputs, JDRaw(text=jd), config, llm=llm, embedder=embedder)
    except (LLMConfigError, EmbeddingConfigError) as exc:
        # 配置错误（如缺少 API key）-> 返回 503 并附带清晰的错误信息。
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    await asyncio.to_thread(store.save, result)  # 设置 run_id + created_at
    return result


@app.get("/runs/{run_id}", response_model=ScreeningResult)
async def get_run(
    run_id: str,
    store: RunStore = Depends(get_store),
) -> ScreeningResult:
    result = await asyncio.to_thread(store.get, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return result


@app.get("/runs")
async def list_runs(
    limit: int = 50,
    store: RunStore = Depends(get_store),
) -> dict:
    runs = await asyncio.to_thread(store.list_runs, limit)
    return {"runs": runs}
