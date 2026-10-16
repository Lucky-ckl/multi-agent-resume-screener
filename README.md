---
title: "CV-Align | Multi-Agent Resume Screener"
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# CV-Align | Multi-Agent Resume Screener

**在线演示：** https://bit.ly/multi_agent-resume-screener

> 基于 **LangGraph** 与 **Gemini** 构建的多智能体简历筛选流水线，通过 **FastAPI** 对外提供服务。

CV-Align 使用一条由多个专职智能体组成的流水线（**简历解析器**、**JD 解析器**、**检索器**、**匹配器**、确定性**评分器**、确定性**健康度检查器**和**评审 agent**）将简历与职位描述进行匹配，内置自我修正循环和完整的审计轨迹以保证可解释性。检索器通过语义（Gemini embedding）搜索从简历中找出与每条 JD 要求最相关的原文片段，让匹配器的打分有据可依——这是一个货真价实的小型 RAG 层，而不是把整份简历一股脑塞给模型。

一套引擎，服务两种角色：

- **候选人模式**："我的简历和这个岗位匹配度如何？该怎么改进？"反馈分为两条轨道：**长期积累**（为该岗位需要培养的技能/经验）和**立即修改**（可以马上完成的确定性简历修改）。
- **招聘者模式**："帮我给这批简历排序，并给出理由。"每位候选人都会附上匹配器的逐板块推理与证据，作为排序的明确依据。

## 为什么采用多智能体设计？

用一次 LLM 调用"给简历打分"是个黑盒：你无法知道候选人为什么排在那个位置，也无法只改进某一个阶段而不波及其他阶段。CV-Align 把这项工作拆分为多个专注的阶段——任务是确定性的就用普通函数，只有需要判断力的环节才用 LLM 智能体：

| 阶段 | 类型 | 职责 |
|------|------|------|
| Parser | agent | 简历 PDF 文本 → 结构化字段（技能、项目、经历） |
| JD Parser | agent | 职位描述 → 结构化要求 |
| Retriever | function | 简历分块 + 按板块检索 top-k 相关原文片段（Gemini embeddings，余弦相似度） |
| Matcher | agent | 基于检索到的原文片段，针对**这份** JD 给出逐板块子分 + 引用证据 |
| Scorer | function | 由子分合成确定性的加权总分 |
| Hygiene | function | 确定性的"立即修改"规则：链接、量化表述、弱动词、过长条目、第一人称、空话套话、通用项目名…… |
| Critic | agent | JD 差距、"长期积累"的技能建设建议、结论，以及一次置信度检查 |

如果 critic 对"分数是否站得住脚"没有把握，会回环到 matcher 重新评估一次（有上限，避免无限循环）。

```
                resume.pdf          职位描述
                    │                     │
                    ▼                     ▼
                 parser ─────────────► jd_parser
                    └──────────┬──────────┘
                               ▼
                          retriever (Gemini embeddings)
                               │
                    ┌────► matcher ──► scorer ──► hygiene ──► critic ─┐
                    │                                                  │
                    └──────── 自我修正（最多重试 1 次） ◄────────────┘
                                          │
                                          ▼
                                 排序后、可解释的筛选结果
```

## 候选人反馈：两条轨道

当你筛选自己的简历（候选人模式）时，反馈被刻意分成两类，因为这两种改进的时间尺度完全不同：

- **长期积累，面向该岗位。** 你尚未明确满足的 JD 要求（`gaps`），加上 critic 给出的前瞻性技能建设建议（`suggestions`）：要学哪些技术、下一步做什么样的项目/积累什么样的经验。这些是你要*逐步成长*去达成的，不是今天改一下就完事的。
- **立即修改，快速改动简历。** 由确定性健康度检查器发现的客观的、基于规则的问题（`hygiene_issues`）：补充量化成效、用强动作动词开头、压缩过长条目、去掉第一人称和空话套话、补上缺失的链接。这些修改即时生效、不依赖 LLM、结果可复现，UI 会在下方单独的板块中展示。

"立即修改"这一类是确定性的，因此成本低、站得住脚（"我们把客观的规则检查与 LLM 判断相结合，而不是一个大 prompt 包打天下"），并且不增加任何延迟。

## 当前状态

✅ 核心引擎 + HTTP API 已完成并通过测试（95 个测试，完全离线运行）。核心流水线已针对 Gemini 实测验证；证据检索层的离线测试使用 fake embedder，尚未针对线上 Gemini embeddings API 实测。完整设计见 `docs/architecture.md`。

## 技术栈

- Python 3.12
- FastAPI + Uvicorn
- LangGraph + LangChain（智能体编排）
- Google Gemini（免费档），提供商做了抽象，可换用 Groq
- pypdf（PDF 文本提取）
- SQLite（运行历史 / 可解释性）

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/ShreyanshMehra/multi-agent-resume-screener.git
cd multi-agent-resume-screener

# 2. 创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
copy .env.example .env        # Windows  (macOS/Linux 使用 `cp`)
# 然后编辑 .env，填入你的免费 Gemini 密钥：
# https://aistudio.google.com/apikey
```

## 使用方法

### 启动 API

```bash
uvicorn multi_agent_resume_screener.api.main:app --reload
# 打开 http://127.0.0.1:8000/docs 查看交互式 Swagger UI
```

将简历与职位描述进行匹配筛选：

```bash
# 招聘者模式：对多份简历排序
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer. Required: Python, Go, PostgreSQL, 3+ years." \
  -F "mode=recruiter" \
  -F "resumes=@alice.pdf" \
  -F "resumes=@bob.pdf"

# 候选人模式：给出改进一份简历的反馈
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer..." \
  -F "mode=candidate" \
  -F "resumes=@my_cv.pdf"

# 之后查询某次已存储的运行结果
curl http://127.0.0.1:8000/runs/<run_id>
```

### 命令行脚本

```bash
python scripts/check_llm.py                       # 验证 LLM 密钥是否可用
python scripts/parse_resume.py resume.pdf         # PDF → 结构化简历
python scripts/screen.py resume.pdf job.txt       # 完整流水线 + 智能体轨迹
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活检查 |
| POST | `/screen` | 将简历 PDF 与 JD 匹配筛选，返回排序结果 |
| GET | `/runs/{id}` | 查询某次已存储的筛选运行 |
| GET | `/runs` | 列出最近的运行记录 |

`/screen` 表单字段：`jd`（文本）、`mode`（`candidate`\|`recruiter`）、
`critic_mode`（`fast`\|`full`）、`critic_top_k`（整数）、`resumes`（PDF 文件，可多个）。

## 测试

```bash
pip install -e ".[dev]"
pytest -q                       # 95 个测试，完全离线（fake LLM + fake embedder）
ruff check src tests scripts    # 代码检查
```

## 项目结构

```
src/multi_agent_resume_screener/
├── state.py         # 共享的 Pydantic 状态 + 对外结果模型
├── settings.py      # 从 .env 读取的类型化配置
├── pdf.py           # 确定性的 PDF → 文本提取
├── llm/             # 提供商抽象的 LLM 客户端（Gemini/Groq）
├── embeddings/      # Gemini embedding 客户端（简历证据检索）
├── agents/          # parser、jd_parser、matcher、critic
├── pipeline/        # scorer、hygiene、retrieval（确定性）、graph、screen
├── api/             # FastAPI 应用
└── storage/         # SQLite 运行持久化
```

## 部署

项目自带 `Dockerfile`（Hugging Face Spaces 使用）和 Render 蓝图（`render.yaml`）。

```bash
# 用 Docker 本地构建并运行
docker build -t multi-agent-resume-screener .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your-key multi-agent-resume-screener
```

**Hugging Face Spaces（推荐，免费）：** `README.md` 的 front matter 已配置为 Docker Space（`app_port: 8000`）。新建一个 **Docker** Space，把本仓库推送上去，并在 Space 的 **secret** 中添加 `GOOGLE_API_KEY`。Space 仅在闲置 48 小时后休眠，所以作品集链接通常都是"温热"可用的。

**Render（免费档）：** 在仓库上创建新的 Blueprint，然后在控制台设置 `GOOGLE_API_KEY`。注意：免费服务闲置 15 分钟后会休眠。两个平台都把 SQLite 文件放在 `/tmp`（临时目录，重新部署后运行历史会清空，对演示来说无所谓）。

## 许可证

[MIT](LICENSE) © 2026 Shreyansh Dutt Mehra
