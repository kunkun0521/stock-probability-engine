"""数据读取、清洗、股票池与涨跌停规则（文档 §3、§4）。

关键工程决策
------------
* 一次读取完整 500 日窗口的**前复权快照**，内部绝不拼接不同次读取的价格（§4.3）。
* 用 pymysql SSCursor 流式把 154 万行填进 float32 矩阵 [股票 × 日期]，
  避免 pandas 逐行构建 154 万个 Decimal 对象造成内存暴涨。
* 所有特征都是 [日期 × 股票] 的 DataFrame，可整块向量化；横截面排名按行计算。
* 停牌日：价格前向填充但 valid=False，样本/收益/模型拟合只使用 valid=True 的格子。
  这是「指标在有效交易日序列上计算」的向量化近似，详见 docs/decisions.md。
* asof 之后的行情根本不会读进内存，天然满足 §13「无未来数据泄漏」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pymysql
from pymysql.cursors import SSCursor

from .db import Db

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "amplitude", "pct_chg", "turnover", "tradestatus", "is_st",
    "pe_ttm", "pb_mrq", "ps_ttm",
]
PERCENT_COLUMNS = {"amplitude", "pct_chg", "turnover"}


@dataclass
class MarketData:
    cfg: dict[str, Any]
    db: Db
    dates: list[date] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)       # f"{code}.{market}"
    markets: dict[str, str] = field(default_factory=dict)
    arr: dict[str, pd.DataFrame] = field(default_factory=dict)
    valid: pd.DataFrame | None = None
    limit: pd.DataFrame | None = None
    universe: list[str] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)
    warn_count: int = 0
    completeness: float = 1.0
    asof: date | None = None
    market_ret: pd.Series | None = None
    market_ret_col: str = "equal_weight"

    # -------------------------------------------------------------- 对外便捷访问
    @property
    def close(self) -> pd.DataFrame:
        return self.arr["close"]

    @property
    def log_close(self) -> pd.DataFrame:
        return self.arr["log_close"]

    @property
    def ret(self) -> pd.DataFrame:
        return self.arr["ret_1"]

    def code_of(self, key: str) -> str:
        return key.split(".")[0]

    def market_of(self, key: str) -> str:
        return self.markets.get(key, "SH")

    def limit_of(self, key: str, d: date) -> float:
        try:
            return float(self.limit.at[d, key])
        except Exception:
            return float(self.cfg["limit_pct"]["main"])

    def history_prices(self, key: str, asof: date, n: int = 60) -> np.ndarray:
        """个股近 n 个有效交易日收盘价，给前端扇形图叠加用。"""
        pos = self.date_pos(asof)
        s = self.arr["close"].iloc[max(0, pos - n + 1): pos + 1][key]
        return s.dropna().to_numpy(dtype=float)

    def date_pos(self, d: date) -> int:
        return self._date_index[d]

    # -------------------------------------------------------------- 主流程
    def load(self, asof: date | None = None, max_stocks: int = 0) -> "MarketData":
        table = self.cfg["db"].get("source_table", "stock_daily")
        flag = int(self.cfg["db"].get("adjustflag", self.cfg.get("adjustflag", 2)))
        lookback = int(self.cfg["lookback_days"])
        with self.db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT DISTINCT trade_date FROM `{table}` "
                            f"WHERE adjustflag=%s ORDER BY trade_date DESC LIMIT %s", (flag, lookback))
                all_dates = [r["trade_date"] for r in cur.fetchall()][::-1]
                if not all_dates:
                    raise RuntimeError(f"{table} 中没有 adjustflag={flag} 的交易日数据")
                self.asof = asof or all_dates[-1]
                if self.asof not in all_dates:
                    # 允许指定历史 asof 做 walk-forward；必须在窗口内
                    raise RuntimeError(f"asof={self.asof} 不在最近 {lookback} 个交易日窗口内")
                self.dates = [d for d in all_dates if d <= self.asof]
                first, last = self.dates[0], self.dates[-1]
                cur.execute(f"SELECT DISTINCT code, market FROM `{table}` "
                            f"WHERE adjustflag=%s AND trade_date BETWEEN %s AND %s "
                            f"ORDER BY code, market", (flag, first, last))
                pairs = [(str(r["code"]), str(r["market"])) for r in cur.fetchall()]
                if max_stocks and max_stocks > 0:
                    pairs = pairs[:max_stocks]
                self.keys = [f"{c}.{m}" for c, m in pairs]
                self.markets = {f"{c}.{m}": m for c, m in pairs}
                key_idx = {k: i for i, k in enumerate(self.keys)}
                self._date_index = {d: j for j, d in enumerate(self.dates)}
                n_k, n_d = len(self.keys), len(self.dates)
                raw = {c: np.full((n_k, n_d), np.nan, dtype=np.float32) for c in NUMERIC_COLUMNS}

                cols = ",".join(NUMERIC_COLUMNS)
                cur.close()
                stream = conn.cursor(SSCursor)
                stream.execute(
                    f"SELECT code,market,trade_date,{cols} FROM `{table}` "
                    f"WHERE adjustflag=%s AND trade_date BETWEEN %s AND %s",
                    (flag, first, last))
                for row in stream:
                    key = f"{row[0]}.{row[1]}"
                    i = key_idx.get(key)
                    j = self._date_index.get(row[2])
                    if i is None or j is None:
                        continue
                    for k, name in enumerate(NUMERIC_COLUMNS):
                        v = row[3 + k]
                        if v is not None:
                            raw[name][i, j] = float(v)
                stream.close()

        # [股票 × 日期] → [日期 × 股票]
        for name in NUMERIC_COLUMNS:
            a = raw[name].T.copy()
            if name in PERCENT_COLUMNS:
                a /= 100.0
            self.arr[name] = pd.DataFrame(a, index=self.dates, columns=self.keys, dtype=np.float32)
        del raw
        self._clean()
        self._build_derived()
        self._build_universe()
        return self

    # -------------------------------------------------------------- §4.2 清洗
    def _clean(self):
        o, h, l, c = (self.arr[x] for x in ("open", "high", "low", "close"))
        ok = (c > 0) & (h >= l) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c))
        self.warn_count += int((~ok & c.notna()).to_numpy().sum())
        for name in ("open", "high", "low", "close", "preclose"):
            self.arr[name] = self.arr[name].where(ok)
        # 停牌：tradestatus=0 或 volume=0
        suspended = (self.arr["tradestatus"].fillna(0) == 0) | (self.arr["volume"].fillna(0) <= 0)
        self.valid = ok & (~suspended)
        self.arr["suspended"] = suspended.astype(np.float32)
        # 估值非正视为缺失，不得填 0
        for name in ("pe_ttm", "pb_mrq", "ps_ttm"):
            self.arr[name] = self.arr[name].where(self.arr[name] > 0)
        # 涨跌停幅度：按板块 + ST 规则，全部来自 config.limit_pct
        lp = self.cfg["limit_pct"]
        st = (self.arr["is_st"].fillna(0) == 1)
        gem = pd.DataFrame(
            np.repeat(np.array([[k.startswith("30") for k in self.keys]], dtype=bool),
                      len(self.dates), axis=0),
            index=self.dates, columns=self.keys)
        lim = pd.DataFrame(float(lp["main"]), index=self.dates, columns=self.keys, dtype=np.float32)
        lim = lim.mask(gem, float(lp["gem"])).mask(st, float(lp["st"]))
        self.limit = lim
        # 前复权价格：停牌日 ffill，仅用于连续计算收益率/指标；样本仍由 valid 过滤
        self.arr["close_filled"] = self.arr["close"].ffill().astype(np.float32)
        self.arr["log_close"] = np.log(self.arr["close_filled"].where(self.arr["close_filled"] > 0))

    def _build_derived(self):
        lc = self.arr["log_close"]
        r = lc.diff()
        # pct_chg 交叉校验（§4.3），差异超过 0.001 计入告警
        implied = np.log1p(self.arr["pct_chg"].clip(lower=-0.999))
        diff = (r - implied).abs()
        self.warn_count += int(((diff > 0.001) & self.valid).fillna(False).to_numpy().sum())
        self.arr["ret_1"] = r.where(self.valid).astype(np.float32)
        # 停牌日收益置 0，但样本仍由 valid 过滤；不参与模型拟合
        self.ret_raw = r.where(self.valid, 0.0).astype(np.float32)
        # market proxy（§3.3）：当日股票池内有效个股收益等权平均
        if str(self.cfg.get("market_proxy", "equal_weight")) == "amount_weighted":
            w = self.arr["amount"].where(self.valid)
            self.market_ret_col = "amount_weighted"
            self.market_ret = (self.arr["ret_1"] * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
        else:
            self.market_ret = self.arr["ret_1"].mean(axis=1)
        self.market_ret = self.market_ret.fillna(0.0)

    # -------------------------------------------------------------- §4.1 股票池
    def _build_universe(self):
        c = self.cfg
        prefixes = tuple(c["universe_prefixes"])
        excluded = {"code_prefix": 0, "st": 0, "min_history": 0, "suspension": 0,
                    "no_asof_data": 0}
        asof = self.asof
        pos = self.date_pos(asof)
        look_susp = int(c["suspension_lookback"])
        # arr / valid 的形状是 [日期 × 股票]，第 0 维是日期、第 1 维是股票
        valid = self.valid.to_numpy(dtype=bool)
        susp = self.arr["suspended"].to_numpy(dtype=float)
        keep = []
        for i, key in enumerate(self.keys):
            code = self.code_of(key)
            if not code.startswith(prefixes):
                excluded["code_prefix"] += 1
                continue
            if c.get("exclude_st", True) and float(self.arr["is_st"].iat[pos, i] or 0) == 1:
                excluded["st"] += 1
                continue
            n_hist = int(valid[:pos + 1, i].sum())
            if n_hist < int(c["min_history"]):
                excluded["min_history"] += 1
                continue
            n_susp = int(np.nansum(susp[max(0, pos - look_susp + 1):pos + 1, i]))
            if n_susp >= int(c["max_suspension_days"]):
                excluded["suspension"] += 1
                continue
            if not valid[pos, i]:
                excluded["no_asof_data"] += 1
                continue
            keep.append(key)
        self.universe = keep
        self.excluded = excluded
        # §4.5 数据完整性：asof 有效行数 / 近 20 日日均有效行数
        n_days = int(c.get("complete_check_days", 20))
        daily = self.valid.iloc[max(0, pos - n_days):pos + 1].sum(axis=1).astype(float)
        base = daily.iloc[:-1].mean() if len(daily) > 1 else daily.iloc[-1]
        self.completeness = float(daily.iloc[-1] / base) if base and base > 0 else 1.0
        if self.completeness < float(c.get("min_valid_row_ratio", 0.9)):
            action = str(c.get("incomplete_action", "WARN")).upper()
            msg = (f"asof={asof} 有效行占比 {self.completeness:.2%} < "
                   f"{float(c['min_valid_row_ratio']):.0%}，当日数据可能未导完")
            if action == "ABORT":
                raise RuntimeError(msg)
            self.warn_count += 1
            self.completeness_message = msg
        else:
            self.completeness_message = ""

    # -------------------------------------------------------------- 给回测/图表用
    def price(self, key: str, d: date, col: str = "close") -> float:
        try:
            v = self.arr[col].at[d, key]
            return float(v) if np.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    def is_limit_up(self, key: str, d: date) -> bool:
        """判定涨停：pct_chg ≥ L-eps 且 close==high（§4.4）。"""
        try:
            pct = float(self.arr["pct_chg"].at[d, key])
            L = self.limit_of(key, d)
            return bool(pct >= L - float(self.cfg["limit_epsilon"]) and
                        abs(float(self.arr["close"].at[d, key]) - float(self.arr["high"].at[d, key])) < 1e-6)
        except Exception:
            return False

    def is_limit_down(self, key: str, d: date) -> bool:
        try:
            pct = float(self.arr["pct_chg"].at[d, key])
            L = self.limit_of(key, d)
            return bool(pct <= -L + float(self.cfg["limit_epsilon"]) and
                        abs(float(self.arr["close"].at[d, key]) - float(self.arr["low"].at[d, key])) < 1e-6)
        except Exception:
            return False
