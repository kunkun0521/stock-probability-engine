"""MySQL 访问层：pymysql 直连，不依赖 SQLAlchemy（本机 WDAC 会拦截 SQLAlchemy 的 C 扩展 .pyd）。

提供：连接解析、幂等建表、分页查询、批量 UPSERT（§10.2 幂等：同一主键重跑覆盖）。
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, unquote, urlparse

import pymysql
from pymysql.cursors import DictCursor

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

# 幂等补充索引：MySQL 不支持 CREATE INDEX IF NOT EXISTS，重复执行报「duplicate key name」时忽略即可。
# 这些是列表/总览聚合的覆盖索引，避免回表读取 components_json（JSON 列很大，回表会让 GROUP BY 变慢十几秒）。
EXTRA_INDEXES = [
    "ALTER TABLE `prob_prediction` ADD INDEX `idx_prob_pred_ov` "
    "(`asof_date`,`pred_type`,`risk_level`,`reliability_grade`,`p_up`,`code`)",
    "ALTER TABLE `prob_prediction` ADD INDEX `idx_prob_pred_oos` "
    "(`pred_type`,`status`,`horizon`,`asof_date`)",
]


def parse_db_url(url: str) -> dict[str, Any]:
    u = urlparse(url)
    if u.scheme not in ("mysql", "mysql+pymysql"):
        raise ValueError(f"不支持的数据库 URL：{u.scheme}")
    qs = parse_qs(u.query)
    return dict(
        host=u.hostname or "localhost",
        port=u.port or 3306,
        user=unquote(u.username or "root"),
        password=unquote(u.password or ""),
        database=(u.path or "/").lstrip("/").split("?")[0] or "stock_eastmoney",
        charset=qs.get("charset", ["utf8mb4"])[0],
    )


def clean_value(v: Any) -> Any:
    """NaN/Inf → None，Decimal/np 标量 → Python 原生，date/datetime 原样交给驱动。"""
    if v is None:
        return None
    if isinstance(v, (float,)) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item") and not isinstance(v, (str, bytes, date, datetime, Decimal)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (dict, list, tuple)):
        import json
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


class Db:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.conn_args = parse_db_url(cfg["db"]["url"])

    def connect(self):
        return pymysql.connect(
            **self.conn_args, cursorclass=DictCursor, autocommit=True,
            connect_timeout=10, read_timeout=300, write_timeout=300,
        )

    def query(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                # params 为空时必须完全不传，否则 pymysql 会对 SQL 里的 % 做字符串格式化
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
                return list(cur.fetchall())

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        rows = self.query(sql, params)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                return cur.execute(sql, params) if params else cur.execute(sql)

    def ensure_schema(self) -> None:
        """执行 schema.sql（全是 CREATE TABLE IF NOT EXISTS，可重复执行）。"""
        # 必须先逐行去掉注释再按分号切分，否则首段（注释 + 第一条 CREATE）会被整段丢掉
        lines = [l for l in SCHEMA_FILE.read_text(encoding="utf-8").splitlines()
                 if not l.strip().startswith("--")]
        statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
        with self.connect() as conn:
            with conn.cursor() as cur:
                for body in statements:
                    cur.execute(body)
                for ddl in EXTRA_INDEXES:
                    try:
                        cur.execute(ddl)
                    except Exception:
                        # 索引已存在（MySQL 1061），忽略即可
                        pass

    def upsert(self, table: str, rows: Iterable[dict[str, Any]],
               columns: Sequence[str], update_cols: Sequence[str] | None = None,
               batch: int = 800) -> int:
        """批量 INSERT ... ON DUPLICATE KEY UPDATE；rows 里缺的列自动写 NULL。"""
        cols = list(columns)
        upd = [c for c in (update_cols or cols) if c not in ("code", "market", "asof_date",
                                                              "horizon", "pred_type", "model_version",
                                                              "run_id", "signal_key", "train_end_date",
                                                              "eval_date", "model_name", "scope",
                                                              "metric_name", "backtest_id", "trade_date",
                                                              "output_name", "bin_no", "id")]
        placeholders = ",".join(["%s"] * len(cols))
        head = f"INSERT INTO `{table}` ({','.join('`' + c + '`' for c in cols)}) VALUES ({placeholders})"
        if upd:
            head += " ON DUPLICATE KEY UPDATE " + ",".join(f"`{c}`=VALUES(`{c}`)" for c in upd)
        buf: list[tuple] = []
        total = 0
        with self.connect() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    buf.append(tuple(clean_value(row.get(c)) for c in cols))
                    if len(buf) >= batch:
                        cur.executemany(head, buf)
                        total += len(buf)
                        buf.clear()
                if buf:
                    cur.executemany(head, buf)
                    total += len(buf)
        return total

    def insert(self, table: str, row: dict[str, Any]) -> int:
        cols = list(row.keys())
        sql = (f"INSERT INTO `{table}` ({','.join('`' + c + '`' for c in cols)}) "
               f"VALUES ({','.join(['%s'] * len(cols))})")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(clean_value(row[c]) for c in cols))
                return int(cur.lastrowid)

    def update(self, table: str, values: dict[str, Any], where: str,
               where_params: Sequence[Any] = ()) -> int:
        sets = ",".join(f"`{k}`=%s" for k in values)
        return self.execute(f"UPDATE `{table}` SET {sets} WHERE {where}",
                            (*[clean_value(v) for v in values.values()], *where_params))

    def table_exists(self, name: str) -> bool:
        return bool(self.scalar(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name=%s", (name,)))
