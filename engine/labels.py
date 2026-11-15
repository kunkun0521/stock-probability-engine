"""标签、purge 与样本切分（文档 §5.1、§8.1）。

R_h(t) = ln(C_{t+h}/C_t)，Y_h(t) = 1[R_h(t)>0]。
标签只在「有效交易日」上生成；前向窗口 (t, t+h] 含停牌日的样本默认剔除。
训练样本必须再经 purge：标签日期不得落进测试块，即 t <= test_start - h - 1。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import MarketData


@dataclass
class LabelFrame:
    horizon: int
    r_h: pd.DataFrame          # 对数收益
    y_h: pd.DataFrame          # 0/1（NaN 表示不可用）
    usable: pd.DataFrame       # 布尔：能作为标签样本

    def at(self, pos: int, keys: list[str]) -> np.ndarray:
        return self.r_h.iloc[pos].reindex(keys).to_numpy(dtype=np.float32)

    def y_at(self, pos: int, keys: list[str]) -> np.ndarray:
        return self.y_h.iloc[pos].reindex(keys).to_numpy(dtype=np.float32)

    def usable_at(self, pos: int, keys: list[str]) -> np.ndarray:
        return self.usable.iloc[pos].reindex(keys).fillna(False).to_numpy(dtype=bool)

    def base_rate(self, positions: list[int], keys: list[str]) -> float:
        vals = [self.y_h.iloc[p].reindex(keys)[self.usable.iloc[p].reindex(keys).fillna(False)]
                for p in positions]
        if not vals:
            return 0.5
        a = pd.concat(vals).dropna().to_numpy(dtype=float)
        return float(a.mean()) if a.size else 0.5


def build_labels(md: MarketData, h: int, cfg: dict[str, Any] | None = None) -> LabelFrame:
    cfg = cfg or md.cfg
    lc = md.arr["log_close"]
    r_h = (lc.shift(-h) - lc).where(md.valid)
    valid_future = md.valid.shift(-h).fillna(False).astype(bool)
    susp = md.arr["suspended"].fillna(0.0)
    future_susp = susp.copy()
    for k in range(1, int(h) + 1):
        future_susp = np.maximum(future_susp, susp.shift(-k).fillna(0.0))
    window_ok = future_susp == 0 if cfg.get("allow_suspension_in_window", False) is False else True
    usable = md.valid & valid_future & np.isfinite(r_h)
    if isinstance(window_ok, pd.DataFrame):
        usable = usable & window_ok
    y_h = (r_h > 0).astype(np.float32).where(usable)
    return LabelFrame(horizon=int(h), r_h=r_h.astype(np.float32),
                      y_h=y_h, usable=usable)


def purge_mask(n_dates: int, test_start: int, h: int, extra: int = 0) -> np.ndarray:
    """训练样本可用区间：[0, test_start-h-extra)；其余必须剔除（§8.1 purge）。"""
    m = np.zeros(n_dates, dtype=bool)
    m[:max(0, test_start - int(h) - int(extra))] = True
    return m


def walkforward_blocks(md: MarketData, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """按窗口内交易日序号划分 L1 训练 / OOS 测试块（§8.1）。

    返回 [{test_start, test_end, train_positions}]；受 walkforward.oos_max_days 与 oos_stride
    限制以控制耗时，支持从 checkpoint 断点续跑。
    """
    wf = cfg["walkforward"]
    n = len(md.dates)
    start = int(wf.get("oos_start_day", 241)) - 1      # 文档序号从 1 开始
    last = n - max(cfg["horizons"]) - 1
    start = max(int(cfg["warmup_days"]), start)
    if start > last:
        start = max(int(cfg["warmup_days"]), last - int(wf.get("oos_max_days", 60)))
    end = min(last, start + int(wf.get("oos_max_days", 60)) - 1)
    stride = max(1, int(wf.get("oos_stride", 5)))
    h_max = max(cfg["horizons"])
    blocks = []
    for s in range(start, end + 1, stride):
        e = min(end, s + stride - 1)
        train = [p for p in range(int(cfg["warmup_days"]), max(0, s - h_max - 1))]
        blocks.append(dict(test_start=s, test_end=e, train_positions=train))
    return blocks


def training_positions(cfg: dict[str, Any], n_dates: int, h: int) -> list[int]:
    """LIVE 训练：截止到 asof-h 之前所有有标签的交易日（§8.1）。"""
    lo = int(cfg.get("warmup_days", 60))
    hi = n_dates - int(h) - 1
    return list(range(lo, max(lo, hi) + 1)) if hi >= lo else []
