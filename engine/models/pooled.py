"""合并模型组：M1 条件概率/贝叶斯、M2 逻辑回归、M5 期望值、M6 分位数回归、M10 融合与校准。

文档 §7.2、§7.4、§7.6、§8.1。合并模型用全市场「股票×日期」样本训练，
特征做横截面标准化；训练统计量（缺失中位数、winsorize 分位、均值/标准差、p0）只能来自训练期。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .. import formulas as F
from ..data import MarketData
from ..features import FeatureStore, signal_columns
from ..labels import LabelFrame
from .base import BaseModel

M2_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "z_ret_1", "z_volume_20", "z_turnover_20", "z_amplitude_20",
    "bias_ma5", "bias_ma20", "bias_ma60", "ma_bull",
    "macd_dif", "macd_hist", "rsi_14", "pos_high20", "pos_high60",
    "atr14_ratio", "vol_20", "vol_60", "amplitude_ma20",
    "vol_ratio_20", "amount_ratio_20", "turnover_ma5",
    "up_streak", "down_streak", "limit_up_cnt_20",
    "pe_pctile", "pb_pctile", "ps_pctile",
    "excess_ret_1", "corr_60", "beta_60",
    "cs_rank_ret_5", "cs_z_ret_5", "cs_rank_vol_20", "cs_rank_rsi14",
    "cs_rank_vol_ratio_20", "cs_rank_bias_20", "cs_rank_pe_pctile",
    "pe_missing", "pb_missing", "ps_missing",
]
M6_FEATURES = [
    "ret_5", "ret_20", "z_ret_1", "bias_ma20", "vol_20", "vol_ratio_20",
    "atr14_ratio", "rsi_14", "pe_pctile", "excess_ret_1",
    "cs_rank_ret_5", "cs_rank_vol_20", "cs_rank_rsi14",
    "cs_rank_vol_ratio_20", "cs_rank_bias_20", "cs_z_ret_5",
    "up_streak", "down_streak",
]


# --------------------------------------------------------------------------- 预处理

@dataclass
class Preprocessor:
    """训练期统计的缺失中位数 / winsor 分位 / z 标准化参数（§4.2、§8.1）。"""
    input_names: list[str] = field(default_factory=list)   # fit 时的完整输入列顺序
    names: list[str] = field(default_factory=list)          # 剔除高缺失率后保留的列
    medians: np.ndarray = field(default_factory=lambda: np.array([]))
    lo: np.ndarray = field(default_factory=lambda: np.array([]))
    hi: np.ndarray = field(default_factory=lambda: np.array([]))
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    std: np.ndarray = field(default_factory=lambda: np.array([]))
    kept: list[bool] = field(default_factory=list)

    def fit(self, X: np.ndarray, names: list[str], cfg: dict[str, Any]):
        X = np.asarray(X, dtype=np.float32)
        self.input_names = list(names)
        miss = np.isnan(X).mean(axis=0)
        self.kept = list(miss <= float(cfg.get("feature_missing_drop_ratio", 0.30)))
        self.names = [n for n, k in zip(names, self.kept) if k]
        X = X[:, np.array(self.kept, dtype=bool)]
        self.medians = np.nanmedian(np.where(np.isnan(X), np.nan, X), axis=0)
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        filled = np.where(np.isnan(X), self.medians, X)
        qlo, qhi = cfg.get("winsor_pct", [0.01, 0.99])
        self.lo, self.hi = np.nanquantile(filled, qlo, axis=0), np.nanquantile(filled, qhi, axis=0)
        clipped = np.clip(filled, self.lo, self.hi)
        self.mean, self.std = clipped.mean(axis=0), np.maximum(clipped.std(axis=0), 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self.kept:
            return X
        X = X[:, np.array(self.kept, dtype=bool)]
        X = np.where(np.isnan(X), self.medians, X)
        X = np.clip(X, self.lo, self.hi)
        return ((X - self.mean) / self.std).astype(np.float32)

    def params(self) -> dict[str, Any]:
        return dict(input_names=self.input_names, names=self.names,
                    kept=self.kept, medians=self.medians.tolist(),
                    winsor_lo=self.lo.tolist(), winsor_hi=self.hi.tolist(),
                    mean=self.mean.tolist(), std=self.std.tolist())


# --------------------------------------------------------------------------- 训练样本

@dataclass
class TrainSamples:
    X: np.ndarray
    feature_names: list[str]
    y: np.ndarray
    r_h: np.ndarray
    sigma_h: np.ndarray
    day: np.ndarray
    weight: np.ndarray
    signals: dict[str, np.ndarray]
    active: list[tuple[str, ...]] | None = None
    p0: float = 0.5
    n_days: int = 0


def build_train_samples(md: MarketData, fs: FeatureStore, lf: LabelFrame,
                        positions: list[int], keys: list[str], cfg: dict[str, Any],
                        feature_names: list[str] | None = None,
                        seed: int = 20260101) -> TrainSamples:
    """构造合并训练样本；剔除标签不可用与前向窗口含停牌的样本，按日等权。"""
    names = [n for n in (feature_names or M2_FEATURES) if n in fs.frames]
    h = lf.horizon
    rng = np.random.default_rng(seed)
    max_samples = int(cfg.get("max_train_samples", 150000))
    per_day = max(50, max_samples // max(1, len(positions)))
    cols = [n for n in names if n in fs.frames]
    blocks_X, blocks_y, blocks_r, blocks_s, blocks_d, blocks_sig, blocks_key = [], [], [], [], [], [], []
    ann = float(cfg.get("trading_days_per_year", 244))
    for pos in positions:
        usable = lf.usable_at(pos, keys)
        if not usable.any():
            continue
        idx = np.flatnonzero(usable)
        if idx.size > per_day:
            idx = np.sort(rng.choice(idx, size=per_day, replace=False))
        ks = [keys[i] for i in idx]
        X = fs.cross_section(pos, cols, ks).to_numpy(dtype=np.float32)
        y = lf.y_h.iloc[pos].reindex(ks).to_numpy(dtype=np.float32)
        r = lf.r_h.iloc[pos].reindex(ks).to_numpy(dtype=np.float32)
        # σ_h：用 20 日已实现波动换算到 h 日（训练期近似，避免对 80 万个样本逐个拟合 GARCH）
        v20 = fs.frames["vol_20"].iloc[pos].reindex(ks).to_numpy(dtype=np.float32) / math.sqrt(ann)
        sigma = np.maximum(v20 * math.sqrt(h), 1e-4)
        sig = signal_columns(fs, pos, ks)
        blocks_X.append(X)
        blocks_y.append(y)
        blocks_r.append(r)
        blocks_s.append(sigma)
        blocks_d.append(np.full(len(ks), pos, dtype=np.int32))
        blocks_sig.append(sig)
        blocks_key.append(ks)
    if not blocks_X:
        return TrainSamples(np.zeros((0, len(cols))), cols, np.array([]), np.array([]),
                            np.array([]), np.array([]), np.array([]), {})
    X = np.concatenate(blocks_X, axis=0)
    y = np.concatenate(blocks_y)
    r = np.concatenate(blocks_r)
    sigma = np.concatenate(blocks_s)
    day = np.concatenate(blocks_d)
    signals = {k: np.concatenate([b[k] for b in blocks_sig]) for k in blocks_sig[0]}
    # 允许特征列含缺失：缺失率 >30% 的列由 Preprocessor 整体剔除，其余用训练期横截面中位数填充（§4.2）
    good = np.isfinite(y) & np.isfinite(r) & np.isfinite(sigma)
    X, y, r, sigma, day = X[good], y[good], r[good], sigma[good], day[good]
    signals = {k: v[good] for k, v in signals.items()}
    # 每个交易日等权（避免股票数量多的日子权重过大）
    _, inv, counts = np.unique(day, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    w = w / w.mean()
    return TrainSamples(X=X, feature_names=cols, y=y, r_h=r, sigma_h=sigma,
                        day=day, weight=w.astype(np.float32), signals=signals,
                        p0=float(y.mean()), n_days=int(np.unique(day).size))


# --------------------------------------------------------------------------- M1 贝叶斯信号

class SignalBayes(BaseModel):
    name = "M1_signal_bayes"

    def __init__(self, cfg: dict[str, Any], horizon: int, train_end_pos: int):
        self.cfg, self.horizon, self.train_end_pos = cfg, int(horizon), int(train_end_pos)
        self.p0 = 0.5
        self.m = float(cfg.get("prior_strength", 20))
        self.half_life = float(cfg.get("half_life", 60))
        self.min_n = float(cfg.get("min_n", 30))
        self.stat: dict[str, dict[str, float]] = {}
        self.signal_names: list[str] = []

    def fit(self, samples: TrainSamples) -> "SignalBayes":
        self.p0 = float(samples.p0)
        self.signal_names = list(samples.signals.keys())
        age = np.maximum(self.train_end_pos - samples.day, 0).astype(float)
        w = F.time_decay_weight(age, self.half_life)
        sig_names = self.signal_names
        active = []
        for i in range(samples.y.size):
            a = tuple(n for n in sig_names if samples.signals[n][i])
            active.append(a)
        samples.active = active
        combos = Counter()
        max_combo = int(self.cfg.get("signal_max_combo", 3))
        active_keys = np.empty(samples.y.size, dtype=object)
        for i, a in enumerate(active):
            key = "+".join(sorted(a)[:max_combo]) if 2 <= len(a) <= max_combo else ""
            active_keys[i] = key
            if key:
                combos[key] += 1
        top_combo = {k for k, _ in combos.most_common(int(self.cfg.get("max_combo_stats", 300)))}
        keys = list(sig_names) + sorted(top_combo)

        for key in keys:
            parts = key.split("+")
            if len(parts) > 1:
                # 组合必须「恰好」由这些信号构成，避免与父信号重复计数（向量化比较）
                mask = active_keys == key
            else:
                mask = samples.signals.get(key)
                if mask is None:
                    continue
            if not mask.any():
                continue
            ww = w[mask]
            n = float(ww.sum())
            k = float((ww * samples.y[mask]).sum())
            if n < 1:
                continue
            p_hat = k / n
            post = F.beta_posterior_mean(self.p0, self.m, n, k)
            lo, hi = F.wilson_interval(k, n)
            self.stat[key] = dict(n=n, k=k, n_eff=n, p_hat=p_hat,
                                  posterior_mean=post, ci_low=lo, ci_high=hi)
        return self

    def predict(self, active_signals: list[str]) -> float:
        """多信号 log-odds 相加；组合样本不足时回退到父级单信号（§7.2 M1）。"""
        parts = sorted(active_signals)[:int(self.cfg.get("signal_max_combo", 3))]
        chosen: list[float] = []
        if len(parts) >= 2:
            combo = "+".join(parts)
            s = self.stat.get(combo)
            if s and s["n_eff"] >= self.min_n:
                chosen.append(s["posterior_mean"])
        if not chosen:
            for p in parts:
                s = self.stat.get(p)
                if s and s["n_eff"] >= self.min_n:
                    chosen.append(s["posterior_mean"])
        if not chosen:
            return float(self.p0)
        return float(F.clip_prob(F.naive_bayes_logodds(self.p0, chosen),
                                 float(self.cfg.get("p_min", 0.05)),
                                 float(self.cfg.get("p_max", 0.95))))

    def params(self) -> dict[str, Any]:
        return dict(p0=self.p0, m=self.m, half_life=self.half_life,
                    min_n=self.min_n, signals={k: v for k, v in list(self.stat.items())[:50]})


# --------------------------------------------------------------------------- M2 逻辑回归

class DirectionLogit(BaseModel):
    name = "M2_logistic"

    def __init__(self, cfg: dict[str, Any], horizon: int):
        self.cfg, self.horizon = cfg, int(horizon)
        self.pre = Preprocessor()
        self.model: LogisticRegression | None = None
        self.best_c = 1.0
        self.coef_names: list[str] = []
        self.cv_brier: list[dict[str, float]] = []

    def fit(self, samples: TrainSamples) -> "DirectionLogit":
        X = self.pre.fit(samples.X, samples.feature_names, self.cfg).transform(samples.X)
        self.coef_names = self.pre.names
        y, w = samples.y.astype(float), samples.weight.astype(float)
        # 带 purge 的时间序列 CV 选 C：按交易日切 3 折，训练只用更早的折，中间剔除 h 日
        days = np.unique(samples.day)
        grid = list(self.cfg.get("l2_c_grid", [0.01, 0.1, 1.0]))
        scores = {c: [] for c in grid}
        if days.size >= 3 * (self.horizon + 5):
            cuts = np.array_split(days, 4)
            for fi in (1, 2, 3):
                test_days = cuts[fi]
                train_mask = samples.day < test_days[0] - self.horizon
                test_mask = np.isin(samples.day, test_days)
                if train_mask.sum() < 200 or test_mask.sum() < 50:
                    continue
                for c in grid:
                    m = LogisticRegression(C=float(c), max_iter=200, solver="lbfgs")
                    m.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
                    p = m.predict_proba(X[test_mask])[:, 1]
                    scores[c].append(F.brier_score(y[test_mask], p))
        self.cv_brier = [dict(C=c, brier=float(np.mean(v))) for c, v in scores.items() if v]
        self.best_c = min(self.cv_brier, key=lambda d: d["brier"])["C"] if self.cv_brier else grid[-1]
        self.model = LogisticRegression(C=float(self.best_c), max_iter=300, solver="lbfgs")
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(X.shape[0], 0.5)
        return F.clip_prob(self.model.predict_proba(self.pre.transform(X))[:, 1], 0.01, 0.99)

    def contributions(self, x_row: np.ndarray) -> list[dict[str, float]]:
        """因子贡献 β_j·x_j，给个股详情页的条形图（§11 页面 2d）。"""
        if self.model is None:
            return []
        z = self.pre.transform(np.asarray(x_row, dtype=np.float32).reshape(1, -1))[0]
        coef = self.model.coef_[0]
        items = [dict(name=n, value=float(c * v)) for n, c, v in zip(self.coef_names, coef, z)]
        return sorted(items, key=lambda d: abs(d["value"]), reverse=True)

    def params(self) -> dict[str, Any]:
        return dict(C=self.best_c, coef=dict(zip(self.coef_names,
                    self.model.coef_[0].tolist())) if self.model is not None else {},
                    intercept=float(self.model.intercept_[0]) if self.model is not None else 0.0,
                    cv_brier=self.cv_brier, preprocessor=self.pre.params())


# --------------------------------------------------------------------------- M5 期望值

class ConditionalExpectation(BaseModel):
    name = "M5_conditional_expectation"

    def __init__(self):
        self.k_up, self.k_dn = 0.8, -0.8
        self.n_up, self.n_dn = 0, 0

    def fit(self, samples: TrainSamples) -> "ConditionalExpectation":
        z = samples.r_h / np.maximum(samples.sigma_h, 1e-6)
        z = np.clip(z, np.quantile(z, 0.01), np.quantile(z, 0.99))
        up, dn = z > 0, z <= 0
        self.k_up = float(z[up].mean()) if up.any() else 0.8
        self.k_dn = float(z[dn].mean()) if dn.any() else -0.8
        self.n_up, self.n_dn = int(up.sum()), int(dn.sum())
        return self

    def predict(self, p_up: float, sigma_h: float) -> float:
        return F.two_way_expectation(float(p_up), self.k_up * float(sigma_h),
                                     self.k_dn * float(sigma_h))

    def params(self) -> dict[str, Any]:
        return dict(k_up=self.k_up, k_dn=self.k_dn, n_up=self.n_up, n_dn=self.n_dn)


# --------------------------------------------------------------------------- M6 分位数回归

class PinballQuantile(BaseModel):
    """向量化 mini-batch SGD 的线性分位数回归（pinball loss，§7.4 M6）。

    选它而非 statsmodels QuantReg / LightGBM 的理由见 docs/decisions.md：
    全程 numpy、无额外依赖、80 万样本 × 5 个分位在数秒内完成，且公式与 §14 pinball 定义完全一致。
    """

    name = "M6_quantile"

    def __init__(self, cfg: dict[str, Any], horizon: int):
        self.cfg, self.horizon = cfg, int(horizon)
        self.taus = np.asarray(cfg["quantile_taus"], dtype=float)
        self.pre = Preprocessor()
        self.W: np.ndarray = np.zeros((0, 0))
        self.b: np.ndarray = np.zeros(0)

    def fit(self, samples: TrainSamples, feature_names: list[str] | None = None) -> "PinballQuantile":
        names = [n for n in (feature_names or M6_FEATURES) if n in samples.feature_names]
        idx = [samples.feature_names.index(n) for n in names]
        X = self.pre.fit(samples.X[:, idx], names, self.cfg).transform(samples.X[:, idx])
        y = (samples.r_h / np.maximum(samples.sigma_h, 1e-6)).astype(np.float32)
        y = np.clip(y, np.quantile(y, 0.01), np.quantile(y, 0.99))
        n, p = X.shape
        self.W = np.zeros((len(self.taus), p), dtype=np.float64)
        self.b = np.repeat(np.quantile(y, self.taus), 1).astype(np.float64)
        rng = np.random.default_rng(int(self.cfg.get("seed_base", 20260101)) + self.horizon)
        lr0 = float(self.cfg.get("quantile_learning_rate", 0.05))
        l2 = float(self.cfg.get("quantile_l2", 0.001))
        epochs = int(self.cfg.get("quantile_epochs", 60))
        batch = 4096
        for ep in range(epochs):
            lr = lr0 / (1.0 + ep * 0.05)
            order = rng.permutation(n)
            for s in range(0, n, batch):
                take = order[s:s + batch]
                xb, yb = X[take], y[take]
                pred = xb @ self.W.T + self.b
                resid = pred - yb[:, None]
                grad = (resid < 0).astype(float) - self.taus[None, :]
                self.W -= lr * (grad.T @ xb / len(take) + l2 * self.W)
                self.b -= lr * grad.mean(axis=0)
        return self

    def _align(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """把任意特征顺序的矩阵对齐到训练时 self.pre.input_names 的完整顺序（含后来被剔除的列），
        缺失列留 NaN；transform() 会按 kept 再过滤一次，因此两边列数必须一致。"""
        pos = {n: i for i, n in enumerate(feature_names)}
        out = np.full((X.shape[0], len(self.pre.input_names)), np.nan, dtype=np.float32)
        for j, n in enumerate(self.pre.input_names):
            if n in pos:
                out[:, j] = X[:, pos[n]]
        return out

    def predict(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """返回标准化收益的 5 个分位 [n,5]；调用方乘 σ_h 再转简单收益。"""
        if self.W.size == 0:
            return np.zeros((X.shape[0], len(self.taus)))
        x = self.pre.transform(self._align(X, feature_names))
        q = x @ self.W.T + self.b
        return np.sort(q, axis=1)

    def predict_named(self, row: dict[str, float], sigma_h: float) -> np.ndarray:
        x = self.pre.transform(np.array([[row.get(n, np.nan) for n in self.pre.input_names]],
                                        dtype=np.float32))
        q = (x @ self.W.T + self.b)[0]
        return F.enforce_quantile_monotonic(F.to_simple_return(q * float(sigma_h)))

    def params(self) -> dict[str, Any]:
        return dict(taus=self.taus.tolist(), coef=self.W.tolist(),
                    intercept=self.b.tolist(), preprocessor=self.pre.params())


# --------------------------------------------------------------------------- M10 融合 + 校准

class FusionModel(BaseModel):
    """Stacking（非负系数逻辑回归）+ 校准（isotonic/Platt）+ 截断（§7.6）。

    元模型只用 L1 模型的**样本外**预测训练；样本外不足时退化为以 M2 为主的 Platt 校准。
    """

    name = "M10_fusion"

    def __init__(self, cfg: dict[str, Any], horizon: int):
        self.cfg, self.horizon = cfg, int(horizon)
        self.w: np.ndarray = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
        self.bias: float = 0.0
        self.calibrator = None
        self.calibration_method = "none"
        self.ece_before = 0.0
        self.ece_after = 0.0
        self.n_oos = 0

    @staticmethod
    def features(components: list[dict[str, float]]) -> np.ndarray:
        return np.array([[float(F.logit(c.get("p_m1", 0.5))),
                          float(F.logit(c.get("p_m2", 0.5))),
                          float(F.logit(c.get("p_markov", 0.5))),
                          float(F.logit(c.get("p_base", 0.5))),
                          float(c.get("vol_pctile_self", 0.5))] for c in components],
                        dtype=float)

    def fit(self, components: list[dict[str, float]], y: np.ndarray) -> "FusionModel":
        if len(components) < 200 or y.size < 200:
            return self
        X, yy = self.features(components), np.asarray(y, dtype=float)
        self.n_oos = int(yy.size)
        n_train = int(yy.size * 0.7)
        Xtr, ytr = X[:n_train], yy[:n_train]
        Xcal, ycal = X[n_train:], yy[n_train:]

        def loss(w):
            z = Xtr @ w[1:] + w[0]
            p = np.asarray(F.sigmoid(z))
            v = -np.mean(ytr * np.log(np.clip(p, 1e-9, 1)) +
                         (1 - ytr) * np.log(np.clip(1 - p, 1e-9, 1)))
            g = Xtr.T @ (p - ytr) / len(ytr)
            return v, np.concatenate(([np.mean(p - ytr)], g))

        res = optimize.minimize(loss, np.array([0.0, 0.2, 0.8, 0.1, 0.1, 0.0]),
                                jac=True, method="L-BFGS-B",
                                bounds=[(-3, 3)] + [(0, 3)] * 5)
        if res.success and np.isfinite(res.fun):
            self.bias, self.w = float(res.x[0]), np.maximum(res.x[1:], 0.0)
        raw_tr = np.asarray(F.sigmoid(Xtr @ self.w + self.bias))
        raw_cal = np.asarray(F.sigmoid(Xcal @ self.w + self.bias))
        self.ece_before = F.expected_calibration_error(ycal, raw_cal,
                                                        int(self.cfg.get("calibration_bins", 10)))
        # 阈值按「样本外样本条数」判断，不是矩阵元素个数（§7.6 第 2 步）
        n_cal = int(ycal.size)
        if n_cal >= int(self.cfg.get("isotonic_min_samples", 5000)):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
            iso.fit(raw_cal, ycal)
            self.calibrator = iso
            self.calibration_method = "isotonic"
        elif n_cal >= 200:
            lr = LogisticRegression(C=1e6, max_iter=300)
            lr.fit(np.asarray(F.logit(raw_cal)).reshape(-1, 1), ycal)
            self.calibrator = lr
            self.calibration_method = "platt"
        pred_cal = self._calibrate(raw_cal)
        self.ece_after = F.expected_calibration_error(ycal, pred_cal,
                                                      int(self.cfg.get("calibration_bins", 10)))
        return self

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.calibration_method == "isotonic" and self.calibrator is not None:
            return np.asarray(self.calibrator.predict(raw))
        if self.calibration_method == "platt" and self.calibrator is not None:
            return np.asarray(self.calibrator.predict_proba(
                np.asarray(F.logit(raw)).reshape(-1, 1))[:, 1])
        return raw

    def predict(self, components: list[dict[str, float]]) -> np.ndarray:
        X = self.features(components)
        raw = np.asarray(F.sigmoid(X @ self.w + self.bias))
        p = self._calibrate(raw)
        return F.clip_prob(p, float(self.cfg.get("p_min", 0.05)),
                           float(self.cfg.get("p_max", 0.95)))

    def predict_one(self, c: dict[str, float]) -> float:
        return float(self.predict([c])[0])

    def params(self) -> dict[str, Any]:
        return dict(w=self.w.tolist(), bias=self.bias,
                    calibration=self.calibration_method, n_oos=self.n_oos,
                    ece_before=self.ece_before, ece_after=self.ece_after)


# --------------------------------------------------------------------------- 组合容器

@dataclass
class PooledModels:
    """每个 horizon 一整套合并模型 + 共用统计量。"""
    horizon: int
    p0: float = 0.5
    pre: Preprocessor | None = None
    m1: SignalBayes | None = None
    m2: DirectionLogit | None = None
    m5: ConditionalExpectation | None = None
    m6: PinballQuantile | None = None
    m10: FusionModel | None = None
    train_end_pos: int = 0
    train_end_date: date | None = None
    n_samples: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
