"""筛选运行的 SQLite 持久化（可解释性 / 审计轨迹）。

使用标准库 ``sqlite3``（无额外依赖）。每次筛选运行以一行的形式存储，
附带完整的 JSON 负载，因此之后可以通过 ``GET /runs/{id}`` 取回某次运行，
也可以列出最近的运行。每次操作都新开一个连接，这让该存储可以安全地
从 FastAPI 的工作线程中调用。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from multi_agent_resume_screener.state import ScreeningResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    job_title   TEXT,
    n_candidates INTEGER NOT NULL,
    result_json TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """一个基于 SQLite 的小型筛选运行存储。"""

    def __init__(self, db_path: str | Path = "runs.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        # ":memory:" 数据库不能跨连接持久化，因此每次操作新建一个连接
        # 会丢失数据；这里要防范这个坑。
        if self.db_path == ":memory:":
            raise ValueError(
                "RunStore needs a file path; ':memory:' is not supported "
                "because each operation opens a new connection."
            )
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def save(self, result: ScreeningResult) -> str:
        """持久化一次筛选结果并返回其 run id。

        如果 ``run_id`` 和 ``created_at`` 尚未填充，会原地修改 ``result``
        来设置它们。
        """
        run_id = result.run_id or uuid4().hex
        created_at = result.created_at or _now_iso()
        result.run_id = run_id
        result.created_at = created_at

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(id, created_at, mode, job_title, n_candidates, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created_at,
                    result.mode,
                    result.job_title,
                    len(result.candidates),
                    result.model_dump_json(),
                ),
            )
        return run_id

    def get(self, run_id: str) -> ScreeningResult | None:
        """按 id 返回存储的运行记录，未找到则返回 ``None``。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ScreeningResult.model_validate_json(row["result_json"])

    def list_runs(self, limit: int = 50) -> list[dict]:
        """按最新优先返回最近运行的摘要。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, mode, job_title, n_candidates "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
