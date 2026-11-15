"""概率引擎的全部数学公式（纯函数、无 IO、无随机性，除显式传入 rng 的函数）。

设计约定
--------
* 这是全引擎唯一的公式实现：模型、评估、回测、单元测试都调用这里，避免多处各写一份导致口径不一致。
* 每个函数注释里的「文档 §x.y」指向《A股股票概率预测引擎需求与设计文档》v1.0 对应章节，
  前端「公式说明」弹窗里的公式、参数来源、入参统计口径与之一一对应。
* 内部统一用**对数收益** R_h = ln(C_{t+h}/C_t)；对外输出统一用**简单收益** exp(R)-1。
* 概率裁剪统一走 clip_prob，保证 0<p<1，避免 logit / log 出现 inf。

依赖：numpy / scipy（系统 Python 3.14 已具备）。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

EPS = 1e-12


# --------------------------------------------------------------------------- 基础工具

def clip_prob(p: float | np.ndarray, lo: float = 0.05, hi: float = 0.95):
    """把概率截断到 [lo, hi]（文档 §7.6 第 3 步）。"""
    return np.clip(np.asarray(p, dtype=float), lo, hi)


def sigmoid(z: float | np.ndarray):
    """logistic 函数 σ(z)=1/(1+e^{-z})（§7.2 M2）。用分段避免 exp 溢出。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: float | np.ndarray):
    """logit(p)=ln(p/(1-p))，输入先做 eps 裁剪。"""
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def to_simple_return(log_ret):
    """对数收益 → 简单收益：exp(R)-1（§4.3）。"""
    return np.expm1(np.asarray(log_ret, dtype=float))


def to_log_return(simple_ret):
    """简单收益 → 对数收益：ln(1+r)。"""
    return np.log1p(np.clip(np.asarray(simple_ret, dtype=float), -0.999999, None))


def winsorize(x: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99):
    """横截面 1%/99% 缩尾（§4.2）。分位点只能来自训练期，调用方负责传入训练统计得到的边界。"""
    a = np.asarray(x, dtype=float)
    lo, hi = np.nanquantile(a, [lo_q, hi_q])
    return np.clip(a, lo, hi)


def time_decay_weight(age: np.ndarray | float, half_life: float):
    """时间衰减权重 w=0.5^(age/half_life)（§7.2 M1 动态更新）。age 单位：交易日。"""
    return np.power(0.5, np.asarray(age, dtype=float) / max(half_life, EPS))


# --------------------------------------------------------------------------- §14 测试基准 / 正态基线

def normal_prob_up(mu: float, sigma: float) -> float:
    """正态基线 P(R_h>0)=Φ(mu/sigma)（§8.3 B1；§14 第一行）。"""
    return float(stats.norm.cdf(mu / max(sigma, EPS)))


def normal_tail_above(mu: float, sigma: float, x: float) -> float:
    """P(R>x)=1-Φ((x-mu)/sigma)。"""
    return float(stats.norm.sf((x - mu) / max(sigma, EPS)))


def normal_tail_below(mu: float, sigma: float, x: float) -> float:
    """P(R<x)=Φ((x-mu)/sigma)。"""
    return float(stats.norm.cdf((x - mu) / max(sigma, EPS)))


def bayes_posterior(p_signal_given_up: float, p_signal_given_dn: float, p0: float) -> float:
    """贝叶斯 P(up|S)=p(S|up)p0 / (p(S|up)p0+p(S|dn)(1-p0))（§7.2、§14）。"""
    a = p_signal_given_up * p0
    b = p_signal_given_dn * (1.0 - p0)
    return float(a / max(a + b, EPS))


def beta_posterior_mean(p0: float, m: float, n: float, k: float) -> float:
    """Beta(m·p0, m·(1-p0)) 先验的后验均值 (k+m·p0)/(n+m)（§7.2 M1、§14）。"""
    return float((k + m * p0) / (n + m))


def wilson_interval(k: float, n: float, z: float = 1.96):
    """信号命中率的 Wilson 置信区间，写入 prob_signal_stat.ci_low/ci_high。"""
    if n <= 0:
        return 0.5, 0.5
    ph = k / n
    den = 1.0 + z * z / n
    center = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(max(ph * (1 - ph) / n + z * z / (4 * n * n), 0.0)) / den
    return float(center - half), float(center + half)


def conditional_expectation(parts: Sequence[tuple[float, float]]) -> float:
    """离散期望 E[X]=Σ p_i·x_i（§14「期望值」）。"""
    return float(sum(p * x for p, x in parts))


def two_way_expectation(p_up: float, m_up: float, m_dn: float) -> float:
    """M5 E[R_h]=p_up·m_up+(1-p_up)·m_dn（§7.4、§14「概率与幅度对比」）。"""
    return float(p_up * m_up + (1.0 - p_up) * m_dn)


def z_score(x: float, mu: float, sigma: float) -> float:
    """Z=(X-μ)/σ（§6、§14）。"""
    return float((x - mu) / max(sigma, EPS))


def naive_bayes_logodds(p0: float, p_signals: Sequence[float]) -> float:
    """多信号 log-odds 相加：logit(p)=logit(p0)+Σ(logit(p_i)-logit(p0))（§7.2 M1）。"""
    z = float(logit(p0))
    for p in p_signals:
        z += float(logit(p)) - float(logit(p0))
    return float(sigmoid(z))


# --------------------------------------------------------------------------- §7.2 M3 马尔可夫 / §7.2 二项

def markov_transition_matrix(states: Sequence[int], n_states: int = 3, shrink: float = 0.0,
                             unconditional: Sequence[float] | None = None) -> np.ndarray:
    """带拉普拉斯平滑的转移矩阵 P_ij=(n_ij+1)/(Σ_j n_ij+3)（§7.2 M3）。

    shrink>0 时按 (1-shrink)·P + shrink·无条件状态分布 做行收缩（一阶马尔可夫性不显著时）。
    """
    counts = np.ones((n_states, n_states), dtype=float)
    s = np.asarray(states, dtype=int)
    if s.size >= 2:
        np.add.at(counts, (s[:-1], s[1:]), 1.0)
    P = counts / counts.sum(axis=1, keepdims=True)
    if shrink > 0 and unconditional is not None:
        u = np.asarray(unconditional, dtype=float)
        u = u / max(u.sum(), EPS)
        P = (1.0 - shrink) * P + shrink * u[None, :]
        P = P / P.sum(axis=1, keepdims=True)
    return P


def markov_independence_pvalue(states: Sequence[int], n_states: int = 3) -> float:
    """对转移计数矩阵做卡方独立性检验（§7.2 M3），p>0.10 说明状态几乎没有记忆。"""
    counts = np.zeros((n_states, n_states), dtype=float)
    s = np.asarray(states, dtype=int)
    if s.size >= 2:
        np.add.at(counts, (s[:-1], s[1:]), 1.0)
    counts += 1.0  # 避免 0 频数
    try:
        chi2, p, _, _ = stats.chi2_contingency(counts)
        return float(p)
    except Exception:
        return 1.0


def markov_step(pi: Sequence[float], P: np.ndarray) -> np.ndarray:
    """π_{k+1}=π_k·P（§14 马尔可夫链）。"""
    return np.asarray(pi, dtype=float) @ np.asarray(P, dtype=float)


def markov_horizon(pi1: Sequence[float], P: np.ndarray, h: int) -> np.ndarray:
    """π_h = π_1·P^(h-1)，h≥1（§7.2 M3）。"""
    pi = np.asarray(pi1, dtype=float)
    for _ in range(max(int(h) - 1, 0)):
        pi = markov_step(pi, P)
    return pi / max(pi.sum(), EPS)


def expected_streak(p_stay: float) -> float:
    """期望连涨/连跌天数 1/(1-P_UU)（§7.2 M3）。"""
    return float(1.0 / max(1.0 - p_stay, EPS))


def binomial_pmf(n: int, k: int, p: float) -> float:
    """P(X=k)=C(n,k)p^k(1-p)^(n-k)（§7.2「伯努利与二项分布」、§14）。"""
    return float(stats.binom.pmf(k, n, p))


def binomial_winrate_pvalue(wins: int, n: int, p0: float = 0.5) -> float:
    """策略胜率显著性：H0 胜率=p0 的双侧二项检验（§7.2、§8.5）。"""
    if n <= 0:
        return 1.0
    return float(stats.binomtest(int(wins), int(n), float(p0), alternative="two-sided").pvalue)


# --------------------------------------------------------------------------- §7.3 M4 波动率

def ewma_variance(r: Sequence[float], lam: float = 0.94) -> float:
    """EWMA 递推 σ²_t=λσ²_{t-1}+(1-λ)r²_{t-1}，返回下一期方差（§7.3 回退）。"""
    a = np.asarray(r, dtype=float)
    if a.size == 0:
        return 0.0
    var = float(np.var(a[: min(20, a.size)]))
    for x in a:
        var = lam * var + (1.0 - lam) * x * x
    return var


def garch_h_variance(omega: float, alpha: float, beta: float, var_next: float, h: int) -> float:
    """h 日累计方差（§7.3）：σ̄²=ω/(1-α-β)，E[σ²_{t+k}]=σ̄²+(α+β)^{k-1}(σ²_{t+1}-σ̄²)，
    σ_h²=Σ_{k=1..h} E[σ²_{t+k}]。§14 基准：ω=.05,α=.05,β=.90,σ²_{t+1}=2 → h=2 时 3.95。
    """
    pers = alpha + beta
    bar = omega / max(1.0 - pers, EPS)
    total = 0.0
    for k in range(1, int(h) + 1):
        total += bar + (pers ** (k - 1)) * (var_next - bar)
    return float(total)


def ewma_h_variance(var_next: float, h: int) -> float:
    """EWMA 回退时 σ_h²=h·σ²_{t+1}（§7.3）。"""
    return float(h) * float(var_next)


def student_t_scale(nu: float) -> float:
    """标准化 t(ν) 的尺度 sqrt((ν-2)/ν)，使方差为 1（§7.3）。"""
    return math.sqrt(max(nu - 2.0, EPS) / nu) if nu > 2.0 else 1.0


def ljung_box_pvalue(sq_resid: Sequence[float], lags: int = 10) -> float:
    """标准化残差平方的 Ljung-Box 检验 p 值（§7.3 诊断）。"""
    a = np.asarray(sq_resid, dtype=float)
    n = a.size
    if n < lags + 5:
        return 1.0
    a = a - a.mean()
    denom = float(np.dot(a, a))
    if denom <= 0:
        return 1.0
    acf = np.array([float(np.dot(a[:-k], a[k:])) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(q, lags))


def jarque_bera_pvalue(resid: Sequence[float]) -> float:
    """Jarque-Bera 正态性检验 p 值（§7.3 诊断）。"""
    try:
        return float(stats.jarque_bera(np.asarray(resid, dtype=float)).pvalue)
    except Exception:
        return 1.0


# --------------------------------------------------------------------------- §7.4 M6/M7 分布

def pinball_loss(u: np.ndarray | float, tau: float) -> float:
    """ρ_τ(u)=u·(τ-1[u<0])（§7.4 M6）。"""
    a = np.asarray(u, dtype=float)
    return float(np.mean(a * (tau - (a < 0).astype(float))))


def gbm_terminal_mean(S0: float, mu: float, sigma: float, T: float) -> float:
    """GBM 解析终值均值 E[S_T]=S0·exp(μT)（§7.4 M7、§14 GBM 基准）。"""
    return float(S0 * math.exp(mu * T))


def gbm_step(S: np.ndarray, mu: float, sigma: float, dt: float, z: np.ndarray) -> np.ndarray:
    """S_{t+Δ}=S_t·exp[(μ-σ²/2)Δ+σ√Δ·Z]，Δ=1/244（§7.4 M7 基线）。"""
    return S * np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)


def clip_limit(simple_ret: np.ndarray, limit: float) -> np.ndarray:
    """逐日简单收益按涨跌停截断到 [-L,+L]（§4.4 / §7.4 M7）。"""
    return np.clip(simple_ret, -limit, limit)


def enforce_quantile_monotonic(q: np.ndarray) -> np.ndarray:
    """分位交叉时做排序修正，保证 q05<=q25<=q50<=q75<=q95（§7.4 M6、§14 性质测试）。"""
    return np.sort(np.asarray(q, dtype=float))


# --------------------------------------------------------------------------- §7.5 风险

def var_cvar_from_samples(log_returns: np.ndarray, alpha: float = 0.95):
    """VaR_α=-Q_{1-α}(R_h)、CVaR_α=-E[R_h|R_h<=Q_{1-α}]，以正数损失率表示（§7.5 M8）。

    入参是模拟或历史的**简单收益**序列；返回一个 (var, cvar)。
    """
    a = np.asarray(log_returns, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    q = float(np.quantile(a, 1.0 - alpha))
    tail = a[a <= q]
    return float(-q), float(-tail.mean() if tail.size else q)


def gpd_var_es(u: float, beta: float, xi: float, n: float, n_u: float, p: float):
    """POT/GPD 尾部分位与期望损失（§7.5 M9、§14 基准）。

    VaR_p = u + (β/ξ)·[((n/N_u)(1-p))^{-ξ} - 1]
    ES_p  = (VaR_p + β - ξ·u)/(1-ξ)，要求 ξ<1。
    """
    if abs(xi) < 1e-8:
        var_p = u + beta * math.log(max(n / max(n_u, EPS) * (1.0 - p), EPS))
    else:
        var_p = u + (beta / xi) * (pow(max(n / max(n_u, EPS) * (1.0 - p), EPS), -xi) - 1.0)
    es_p = (var_p + beta - xi * u) / max(1.0 - xi, EPS)
    return float(var_p), float(es_p)


def max_drawdown(prices: Sequence[float]) -> float:
    """MDD=max_t (Peak_t-Price_t)/Peak_t（§7.5，仅历史描述）。"""
    a = np.asarray(prices, dtype=float)
    if a.size < 2:
        return 0.0
    peak = np.maximum.accumulate(a)
    return float(np.max((peak - a) / np.maximum(peak, EPS)))


def annualized_sharpe(daily_returns: Sequence[float], rf: float = 0.02,
                      trading_days: int = 244) -> float:
    """年化 Sharpe=(E[R]-rf)/σ·sqrt(N)（§7.5），rf 为年化无风险利率。"""
    a = np.asarray(daily_returns, dtype=float)
    if a.size < 2:
        return 0.0
    sd = float(np.std(a, ddof=1))
    if sd <= 0:
        return 0.0
    return float((a.mean() - rf / trading_days) / sd * math.sqrt(trading_days))


def kupiec_pvalue(breaches: int, n: int, alpha: float = 0.95) -> float:
    """VaR 突破率的 Kupiec POF 似然比检验 p 值（§8.2、§8.4）。"""
    if n <= 0:
        return 1.0
    p = 1.0 - alpha
    x = min(max(int(breaches), 0), n)
    ph = x / n
    # 直接算对数似然，避免 pow(0.05, 3840) 下溢成 0 再取 log
    ll0 = (n - x) * math.log(max(1.0 - p, 1e-12)) + x * math.log(max(p, 1e-12))
    ll1 = ((n - x) * math.log(max(1.0 - ph, 1e-12))
           + x * math.log(max(ph, 1e-12)))
    lr = max(0.0, 2.0 * (ll1 - ll0))
    return float(stats.chi2.sf(lr, 1))


def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE = mean(ln(σ²_forecast) + σ²_realized/σ²_forecast)（§7.7 波动率择优）。"""
    rv = np.maximum(np.asarray(realized_var, dtype=float), EPS)
    fv = np.maximum(np.asarray(forecast_var, dtype=float), EPS)
    return float(np.mean(np.log(fv) + rv / fv))


# --------------------------------------------------------------------------- §7.6 / §8.2 评估

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Brier = mean((p-y)^2)（§8.2）。"""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    """LogLoss = -mean(y·ln p + (1-y)·ln(1-p))（§8.2）。"""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill_score(y: np.ndarray, p: np.ndarray, p_base: float) -> float:
    """BSS = 1 - Brier_model/Brier_base（§8.2、§8.4）。"""
    base = brier_score(y, np.full_like(np.asarray(y, dtype=float), p_base))
    return float(1.0 - brier_score(y, p) / max(base, EPS))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE：10 等分箱加权 |mean_pred - freq_up|（§8.2）。"""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            ece += m.sum() / p.size * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def rank_ic(pred: np.ndarray, realized: np.ndarray) -> float:
    """横截面 Spearman 秩相关（§8.2 Rank IC）。"""
    a = np.asarray(pred, dtype=float)
    b = np.asarray(realized, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return 0.0
    ra = stats.rankdata(a[m])
    rb = stats.rankdata(b[m])
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def block_bootstrap_ci(values: Sequence[float], block: int = 5, n_boot: int = 400,
                       seed: int = 20260101, alpha: float = 0.05):
    """按交易日 block bootstrap 给指标置信区间（§8.2 显著性）。"""
    a = np.asarray(values, dtype=float)
    if a.size < 2 * block:
        m = float(a.mean()) if a.size else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(a.size / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, a.size - block + 1, size=n_blocks)
        sample = np.concatenate([a[s:s + block] for s in starts])[:a.size]
        means.append(float(sample.mean()))
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def risk_level_from_pctile(pct: float, low: float = 0.30, mid: float = 0.70,
                           high: float = 0.90) -> int:
    """风险等级：max(vol_h 分位, cvar95 分位) 落区间 → 1低/2中/3高/4极高（§7.5）。"""
    if pct < low:
        return 1
    if pct < mid:
        return 2
    if pct < high:
        return 3
    return 4


def grade_from_period(bss: float, ci_low: float, downgrades: int = 0) -> str:
    """可靠性评级（§8.4）：周期层 BSS>0 且 CI 下界>0 → A；BSS>0 → B；否则 C；个股每命中一项降一档，最低 D。"""
    base = "A" if (bss > 0 and ci_low > 0) else ("B" if bss > 0 else "C")
    order = "ABCD"
    return order[min(order.index(base) + downgrades, 3)]


# --------------------------------------------------------------------------- §8.5 回测

def backtest_trade_return(buy_price: float, sell_price: float, commission: float,
                          stamp_duty: float, slippage: float) -> float:
    """单笔净收益（§8.5）：买入价含佣金+滑点，卖出价扣佣金+印花税+滑点。"""
    if buy_price <= 0:
        return 0.0
    cost_in = buy_price * (1 + commission + slippage)
    revenue = sell_price * (1 - commission - stamp_duty - slippage)
    return float(revenue / cost_in - 1.0)


def portfolio_metrics(nav: Sequence[float], trading_days: int = 244, rf: float = 0.02):
    """净值序列 → 年化收益/波动/Sharpe/最大回撤（§8.5）。"""
    a = np.asarray(nav, dtype=float)
    if a.size < 2:
        return dict(annualReturn=0.0, annualVol=0.0, sharpe=0.0, maxDrawdown=0.0)
    r = np.diff(a) / np.maximum(a[:-1], EPS)
    years = a.size / trading_days
    ann_ret = float(a[-1] / max(a[0], EPS)) ** (1.0 / max(years, EPS)) - 1.0
    ann_vol = float(np.std(r, ddof=1) * math.sqrt(trading_days)) if r.size > 1 else 0.0
    sharpe = float((r.mean() - rf / trading_days) / max(np.std(r, ddof=1), EPS)
                   * math.sqrt(trading_days)) if r.size > 1 else 0.0
    return dict(annualReturn=float(ann_ret), annualVol=ann_vol,
                sharpe=sharpe, maxDrawdown=max_drawdown(a))
