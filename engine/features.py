"""特征工程（文档 §6）+ 离散信号注册表。

全部特征形状为 [交易日 × 股票] 的 float32 DataFrame，只用 t 及以前的数据：
rolling / ewm 一律不含未来值，绝不使用 center=True。
横截面特征按「当日全市场」计算排名与 z 值，使模型跨股票、跨行情可比。
features.yaml 记录每个特征的公式、回看天数、入参来源与统计口径，前端公式弹窗同源展示。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import load_features
from .data import MarketData


def _rolling_z(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    mu = df.rolling(n, min_periods=max(3, n // 2)).mean()
    sd = df.rolling(n, min_periods=max(3, n // 2)).std()
    return ((df - mu) / sd.replace(0, np.nan)).astype(np.float32)


def _streaks(flags: np.ndarray) -> np.ndarray:
    """沿时间轴累计连续 True 的长度（向量化到股票维度）。"""
    out = np.zeros(flags.shape, dtype=np.float32)
    cur = np.zeros(flags.shape[1], dtype=np.float32)
    for t in range(flags.shape[0]):
        cur = np.where(flags[t], cur + 1.0, 0.0)
        out[t] = cur
    return out


def _rolling_pctile(df: pd.DataFrame, n: int = 250) -> pd.DataFrame:
    """自身窗口内的历史分位（估值分位用）。pandas rolling.rank 返回当前值在窗口内的百分位。"""
    try:
        return df.rolling(n, min_periods=max(20, n // 5)).rank(pct=True).astype(np.float32)
    except Exception:
        # 老版本 pandas 无 rolling.rank 时退化为 expanding rank（仍只用过去）
        return df.expanding(min_periods=20).rank(pct=True).astype(np.float32)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True).astype(np.float32)


def _cs_z(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).astype(np.float32)


@dataclass
class FeatureStore:
    md: MarketData
    cfg: dict[str, Any]
    registry: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    signals: dict[str, pd.DataFrame] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)

    def frame(self, name: str) -> pd.DataFrame:
        return self.frames[name]

    def cross_section(self, pos: int, cols: list[str] | None = None,
                      keys: list[str] | None = None) -> pd.DataFrame:
        """某交易日的 [股票 × 特征] 截面，供预测用。"""
        use = cols or self.names
        ks = keys or self.md.universe
        return pd.DataFrame({c: self.frames[c].iloc[pos].reindex(ks).to_numpy()
                             for c in use}, index=ks).astype(np.float32)

    def samples(self, positions: list[int], keys: list[str] | None = None,
                cols: list[str] | None = None):
        """把若干交易日的截面拉直成训练用的列式数组（避免 pandas.stack 的开销）。"""
        arrs = []
        use = cols or self.names
        for p in positions:
            arrs.append(self.cross_section(p, use, keys).to_numpy(dtype=np.float32))
        X = np.concatenate(arrs, axis=0) if arrs else np.zeros((0, len(use)), dtype=np.float32)
        return X, use


def build_features(md: MarketData, cfg: dict[str, Any],
                   registry: dict[str, Any] | None = None) -> FeatureStore:
    fs = FeatureStore(md=md, cfg=cfg, registry=registry or load_features())
    f: dict[str, pd.DataFrame] = fs.frames
    c = md.arr
    ret = c["ret_1"]
    close = c["close_filled"]
    high, low = c["high"].ffill(), c["low"].ffill()
    ann = float(cfg["trading_days_per_year"])

    # ---- 收益
    lc = c["log_close"]
    for k in (1, 3, 5, 10, 20):
        f[f"ret_{k}"] = (lc.diff(k)).where(md.valid).astype(np.float32)

    # ---- Z-Score
    f["z_ret_1"] = _rolling_z(ret, 20)
    f["z_volume_20"] = _rolling_z(c["volume"], 20)
    f["z_turnover_20"] = _rolling_z(c["turnover"], 20)
    f["z_amplitude_20"] = _rolling_z(c["amplitude"], 20)

    # ---- 趋势
    mas = {}
    for k in (5, 10, 20, 60):
        mas[k] = close.rolling(k, min_periods=k).mean()
        f[f"bias_ma{k}"] = (close / mas[k].replace(0, np.nan) - 1.0).astype(np.float32)
    f["ma_bull"] = ((mas[5] > mas[10]) & (mas[10] > mas[20]) & (mas[20] > mas[60])) \
        .astype(np.float32)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    f["macd_dif"] = dif.astype(np.float32)
    f["macd_dea"] = dea.astype(np.float32)
    f["macd_hist"] = (2.0 * (dif - dea)).astype(np.float32)
    prev_gap = (dif - dea).shift(1)
    f["macd_gold"] = ((dif > dea) & (prev_gap <= 0)).astype(np.float32)
    f["macd_dead"] = ((dif < dea) & (prev_gap >= 0)).astype(np.float32)

    # ---- 超买超卖
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = (100.0 * gain / (gain + loss).replace(0, np.nan)).fillna(50.0).astype(np.float32)
    for k in (20, 60):
        hh = high.rolling(k, min_periods=max(3, k // 3)).max()
        ll = low.rolling(k, min_periods=max(3, k // 3)).min()
        pos = ((close - ll) / (hh - ll).replace(0, np.nan)).clip(0, 1).fillna(0.5)
        f[f"pos_high{k}"] = pos.astype(np.float32)
        f[f"pos_low{k}"] = (1.0 - pos).astype(np.float32)

    # ---- 波动
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()]) \
        .groupby(level=0).max()
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr14_ratio"] = (atr14 / close.replace(0, np.nan)).astype(np.float32)
    f["vol_20"] = (ret.rolling(20, min_periods=10).std() * np.sqrt(ann)).astype(np.float32)
    f["vol_60"] = (ret.rolling(60, min_periods=30).std() * np.sqrt(ann)).astype(np.float32)
    f["amplitude_ma20"] = c["amplitude"].rolling(20, min_periods=10).mean().astype(np.float32)

    # ---- 量能
    v20 = c["volume"].shift(1).rolling(20, min_periods=10).mean()
    a20 = c["amount"].shift(1).rolling(20, min_periods=10).mean()
    f["vol_ratio_20"] = (c["volume"] / v20.replace(0, np.nan)).astype(np.float32)
    f["amount_ratio_20"] = (c["amount"] / a20.replace(0, np.nan)).astype(np.float32)
    f["turnover_ma5"] = c["turnover"].rolling(5, min_periods=3).mean().astype(np.float32)

    # ---- 连续状态
    up = (ret > 0).fillna(False).to_numpy(dtype=bool)
    dn = (ret < 0).fillna(False).to_numpy(dtype=bool)
    f["up_streak"] = pd.DataFrame(np.minimum(_streaks(up), 20.0),
                                  index=md.dates, columns=md.keys)
    f["down_streak"] = pd.DataFrame(np.minimum(_streaks(dn), 20.0),
                                    index=md.dates, columns=md.keys)
    eps = float(cfg["limit_epsilon"])
    lu = ((c["pct_chg"] >= md.limit - eps) & (c["close"] == c["high"])).astype(np.float32)
    ld = ((c["pct_chg"] <= -(md.limit - eps)) & (c["close"] == c["low"])).astype(np.float32)
    f["limit_up_cnt_20"] = lu.rolling(20, min_periods=1).sum().astype(np.float32)
    f["limit_dn_cnt_20"] = ld.rolling(20, min_periods=1).sum().astype(np.float32)

    # ---- 估值分位（非正/缺失不填 0）
    for src, name in (("pe_ttm", "pe_pctile"), ("pb_mrq", "pb_pctile"), ("ps_ttm", "ps_pctile")):
        f[name] = _rolling_pctile(c[src], int(cfg.get("value_pctile_window", 250)))

    # ---- 市场联动
    m = md.market_ret
    def broadcast(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(s.to_numpy(dtype=np.float32)[:, None], len(md.keys), axis=1),
                            index=md.dates, columns=md.keys)
    f["market_ret_1"] = broadcast(m)
    f["market_ret_5"] = broadcast(m.rolling(5, min_periods=1).sum())
    f["excess_ret_1"] = ret.sub(m, axis=0).astype(np.float32)
    # 60 日滚动 corr / beta（与 market proxy 的配对样本）
    win = 60
    rv = ret.to_numpy(dtype=np.float32)
    mv = m.to_numpy(dtype=np.float32)
    corr = np.full(rv.shape, np.nan, dtype=np.float32)
    beta = np.full(rv.shape, np.nan, dtype=np.float32)
    for t in range(win - 1, rv.shape[0]):
        x = rv[t - win + 1:t + 1]
        y = mv[t - win + 1:t + 1]
        ok = np.isfinite(x) & np.isfinite(y[:, None])
        xm = np.where(ok, x, np.nan)
        xb = np.nanmean(xm, axis=0)
        yb = float(np.nanmean(np.where(np.isfinite(x).any(axis=1), y, np.nan)))
        cov = np.nanmean((xm - xb) * (y[:, None] - yb), axis=0)
        sx = np.nanstd(xm, axis=0)
        sy = float(np.nanstd(y[t - win + 1:t + 1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr[t] = cov / (sx * sy)
            beta[t] = cov / (sy * sy)
    f["corr_60"] = pd.DataFrame(np.nan_to_num(corr, nan=0.0), index=md.dates, columns=md.keys)
    f["beta_60"] = pd.DataFrame(np.nan_to_num(beta, nan=1.0), index=md.dates, columns=md.keys)

    # ---- 横截面（只用当日已算出的特征）
    for src, name in (("ret_5", "cs_rank_ret_5"), ("vol_20", "cs_rank_vol_20"),
                      ("rsi_14", "cs_rank_rsi14"), ("vol_ratio_20", "cs_rank_vol_ratio_20"),
                      ("bias_ma20", "cs_rank_bias_20"), ("pe_pctile", "cs_rank_pe_pctile")):
        f[name] = _cs_rank(f[src])
    f["cs_z_ret_5"] = _cs_z(f["ret_5"])

    # ---- 缺失标志（估值类保留）
    f["pe_missing"] = c["pe_ttm"].isna().astype(np.float32)
    f["pb_missing"] = c["pb_mrq"].isna().astype(np.float32)
    f["ps_missing"] = c["ps_ttm"].isna().astype(np.float32)

    fs.names = [x["name"] for x in fs.registry["features"] if x["name"] in f] + \
               [n for n in ("pe_missing", "pb_missing", "ps_missing") if n in f]

    # ---- M1 离散信号
    sig: dict[str, pd.DataFrame] = fs.signals
    sig["sig_macd_gold"] = f["macd_gold"] == 1
    sig["sig_macd_dead"] = f["macd_dead"] == 1
    sig["sig_rsi_low"] = f["rsi_14"] < 30
    sig["sig_rsi_high"] = f["rsi_14"] > 70
    sig["sig_vol_spike"] = f["vol_ratio_20"] > 2
    sig["sig_ma_bull"] = f["ma_bull"] == 1
    sig["sig_down3"] = f["down_streak"] >= 3
    sig["sig_up3"] = f["up_streak"] >= 3
    sig["sig_limit_up"] = f["limit_up_cnt_20"] > 0
    sig["sig_z_pos"] = f["z_ret_1"] > 2
    sig["sig_z_neg"] = f["z_ret_1"] < -2
    sig["sig_mkt_up"] = f["market_ret_1"] > 0
    return fs


def signal_columns(fs: FeatureStore, pos: int, keys: list[str]) -> dict[str, np.ndarray]:
    """某日各信号的布尔列（[股票] 一维），供 M1 统计与预测使用。"""
    return {k: df.iloc[pos].reindex(keys).fillna(False).to_numpy(dtype=bool)
            for k, df in fs.signals.items()}
