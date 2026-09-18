"""
验收快照存储。

Render 的 Web / Cron 磁盘互不相通，重新部署还会清空。
有 DATABASE_URL 时写入 Postgres；否则只写本地 jsonl，并合并仓库里的种子文件。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SEED_PATH = Path(__file__).resolve().parent / "validation_snapshots.jsonl"


def _read_jsonl(path: Path) -> List[Dict]:
    if not path or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pg_url() -> Optional[str]:
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        return None
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def persistence_backend() -> str:
    return "postgres" if _pg_url() else "file"


def _load_postgres() -> List[Dict]:
    url = _pg_url()
    if not url:
        return []
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_snapshots (
                        snapshot_date TEXT PRIMARY KEY,
                        body JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute("SELECT body FROM validation_snapshots ORDER BY snapshot_date")
                rows = []
                for (body,) in cur.fetchall():
                    if isinstance(body, str):
                        rows.append(json.loads(body))
                    elif isinstance(body, dict):
                        rows.append(body)
            conn.commit()
            return rows
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[snapshot] postgres 读取失败，回退文件: {e}")
        return []


def _upsert_postgres(row: Dict) -> bool:
    url = _pg_url()
    if not url or not row.get("date"):
        return False
    try:
        import psycopg2
        from psycopg2.extras import Json
        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_snapshots (
                        snapshot_date TEXT PRIMARY KEY,
                        body JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO validation_snapshots (snapshot_date, body, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (snapshot_date) DO UPDATE
                    SET body = EXCLUDED.body, updated_at = NOW()
                    """,
                    (row["date"], Json(row)),
                )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[snapshot] postgres 写入失败: {e}")
        return False


def load_rows(local_path: Optional[Path] = None) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for row in _read_jsonl(SEED_PATH):
        if row.get("date"):
            merged[row["date"]] = row
    if local_path is not None:
        for row in _read_jsonl(Path(local_path)):
            if row.get("date"):
                merged[row["date"]] = row
    for row in _load_postgres():
        if row.get("date"):
            merged[row["date"]] = row
    return [merged[k] for k in sorted(merged)]


def upsert_row(row: Dict, local_path: Path) -> str:
    """同日覆盖。返回实际写入后端：postgres 或 file。"""
    if not row.get("date"):
        raise ValueError("snapshot 缺少 date")
    rows = [r for r in load_rows(local_path) if r.get("date") != row["date"]]
    rows.append(row)
    rows.sort(key=lambda r: r.get("date") or "")
    _write_jsonl(Path(local_path), rows)
    if _upsert_postgres(row):
        return "postgres"
    return "file"
