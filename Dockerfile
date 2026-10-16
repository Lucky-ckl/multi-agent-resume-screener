FROM python:3.12-slim

# 不写入 .pyc 文件；立即刷新 stdout/stderr。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DB_PATH=/tmp/runs.db

WORKDIR /app

# 先安装依赖，以获得更好的层缓存。
# 包含 [groq] extra，使镜像内可使用 Groq 提供商。
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[groq]"

EXPOSE 8000

# 使用 shell 形式，以便展开 $PORT（由大多数 PaaS 主机设置，如 Render）。
CMD uvicorn multi_agent_resume_screener.api.main:app --host 0.0.0.0 --port ${PORT}
