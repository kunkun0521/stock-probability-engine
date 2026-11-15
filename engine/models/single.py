"""单股模型：M3 马尔可夫、M4 GARCH(1,1)-t / EWMA、M7 蒙特卡洛(FHS/GBM)、M8 VaR/CVaR、M9 EVT。

文档 §7.2(M3)、§7.3、§7.4(M7)、§7.5。每个模型只用该股窗口内的数据，参数很少。
GARCH 的方差递推是线性的，用 scipy.signal.lfilter 向量化，避免 500 步 Python 循环；
MLE 用 Standardized Student-t 似然，不收敛 / 触边 / α+β≥0.999 一律回退 EWMA。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
from scipy import optimize, signal, stats

from .. import formulas as F
from ..data import MarketData
from .base import BaseModel


# --------------------------------------------------------------------------- M4 波动率

@dataclass
class VolState:
    model: str = "EWMA"
    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    nu: float = 8.0
    loglik: float = 0.0
    converged: bool = False
    n_obs: int = 0
    lb_pvalue: float = 1.0
    jb_pvalue: float = 1.0
    sigma_1d_next: float = 0.02       # 日对数收益标准差
    sigma_daily: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    var_by_h: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    residual_pool: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    vol_pctile_self: float = 0.5
    vol_pctile_mkt: float = 0.5


def _garch_variance(omega: float, alpha: float, beta: float, r: np.ndarray) -> tuple[np.ndarray, float]:
    """σ²_t = ω + α r²_{t-1} + β σ²_{t-1} 的向量化解；返回 (样本内 σ², σ²_{t+1})。"""
    init = float(np.var(r[:min(30, r.size)])) or 1e-6
    u = omega + alpha * (r ** 2)
    u = np.concatenate(([init], u[:-1]))
    u[0] = init
    v = signal.lfilter([1.0], [1.0, -max(min(beta, 0.9995), 0.0)], u)
    v = np.maximum(v, 1e-12)
    var_next = float(omega + alpha * r[-1] ** 2 + beta * v[-1])
    return v, var_next


def fit_volatility(r: np.ndarray, cfg: dict[str, Any], max_h: int,
                   force_ewma: bool = False) -> VolState:
    """M4：GARCH(1,1)-t（收益乘 100 拟合再还原），失败回退 EWMA（§7.3）。

    入参 r：该股**有效交易日**的对数收益序列（已剔除停牌日），按时间升序，最近 500 个。
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < max(60, int(cfg.get("garch_min_obs", 250)) // 2):
        return VolState(n_obs=int(r.size))
    r = r[-500:]
    scaled = r * 100.0
    init = float(np.var(scaled[:30])) or 1e-6
    st = VolState(n_obs=int(r.size))
    lam = float(cfg.get("ewma_lambda", 0.94))

    if not force_ewma and r.size >= int(cfg.get("garch_min_obs", 250)):
        def neg_ll(theta):
            om, al, be, nu = theta
            if al + be >= 0.9995 or om <= 0:
                return 1e12
            v, _ = _garch_variance(om, al, be, scaled)
            scale = math.sqrt(max(nu - 2.0, 1e-6) / nu)
            sig = np.sqrt(v) * scale
            z = scaled / np.maximum(sig, 1e-8)
            ll = stats.t.logpdf(z, df=nu) - np.log(np.maximum(sig, 1e-8))
            return float(-np.sum(ll)) if np.isfinite(ll).all() else 1e12

        x0 = np.array([init * 0.05, 0.08, 0.88, 8.0])
        bounds = [(1e-6, max(init * 2.0, 1e-3)), (1e-6, 0.45), (0.05, 0.994), (2.5, 30.0)]
        try:
            res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                                    options=dict(maxiter=int(cfg.get("garch_max_iter", 120))))
            om, al, be, nu = res.x
            ok = bool(res.success and np.isfinite(res.fun) and om > 1e-6
                      and al + be < 0.999 and 2.5 < nu <= 30.0
                      and al > 1e-5 and be < 0.993 and om < bounds[0][1] * 0.98)
            if ok:
                v, var_next = _garch_variance(om, al, be, scaled)
                s = math.sqrt(max(nu - 2, 1e-6) / nu)
                z = scaled / (np.sqrt(v) * s)
                st = VolState(
                    model="GARCH", omega=float(om), alpha=float(al), beta=float(be),
                    nu=float(nu), loglik=float(-res.fun), converged=True, n_obs=int(r.size),
                    lb_pvalue=F.ljung_box_pvalue(z ** 2, 10), jb_pvalue=F.jarque_bera_pvalue(z),
                    sigma_1d_next=math.sqrt(max(var_next, 1e-12)) / 100.0 * s,
                    residual_pool=((z - z.mean()) / max(z.std(), 1e-8)).astype(np.float32),
                )
                st.var_by_h = np.array(
                    [F.garch_h_variance(om, al, be, var_next, h) / 10000.0
                     for h in range(1, int(max_h) + 1)], dtype=float)
        except Exception:
            st = VolState(n_obs=int(r.size))

    if st.model != "GARCH":
        var_next = F.ewma_variance(scaled, lam) / 10000.0
        st.model = "EWMA"
        st.nu = 8.0
        st.loglik = 0.0
        st.converged = False
        st.sigma_1d_next = math.sqrt(max(var_next, 1e-12))
        st.var_by_h = np.array([F.ewma_h_variance(var_next, h)
                                for h in range(1, int(max_h) + 1)], dtype=float)
        st.lb_pvalue = F.ljung_box_pvalue((r / max(st.sigma_1d_next, 1e-8)) ** 2, 10)
        st.jb_pvalue = F.jarque_bera_pvalue(r / max(st.sigma_1d_next, 1e-8))
        z = r / max(st.sigma_1d_next, 1e-8)
        st.residual_pool = ((z - z.mean()) / max(z.std(), 1e-8)).astype(np.float32)
        st.n_obs = int(r.size)

    # 逐日条件波动（FHS 用）：由 h 日累计方差反推每日增量，恒正
    inc = np.diff(np.concatenate(([0.0], st.var_by_h)))
    st.sigma_daily = np.sqrt(np.maximum(inc, 1e-12)).astype(np.float32)
    # 自身历史条件波动分位（用 20 日已实现波动序列近似）
    rv20 = np.array([np.std(r[max(0, t - 19):t + 1]) for t in range(20, r.size, 5)])
    st.vol_pctile_self = float((rv20 < st.sigma_1d_next).mean()) if rv20.size else 0.5
    return st


# --------------------------------------------------------------------------- M3 马尔可夫

@dataclass
class MarkovState:
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3) / 3)
    current: str = "F"
    pvalue: float = 1.0
    pi_h: dict[int, np.ndarray] = field(default_factory=dict)
    p_uu: float = 1 / 3
    p_dd: float = 1 / 3
    exp_up: float = 1.5
    exp_dn: float = 1.5
    p_markov_h: dict[int, float] = field(default_factory=dict)


def fit_markov(r: np.ndarray, cfg: dict[str, Any], horizons: list[int],
               sigma_hist: np.ndarray | None = None) -> MarkovState:
    """3 状态（U/F/D）马尔可夫链，θ_t=κ·σ_t 自适应阈值，拉普拉斯平滑 + 卡方收缩（§7.2 M3）。

    σ_t 优先用 M4 的条件波动；不可用时用 20 日滚动标准差（文档允许）。
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        return MarkovState()
    fallback = np.nanstd(r)
    if sigma_hist is None or np.asarray(sigma_hist).size != r.size:
        import pandas as pd
        sigma_hist = pd.Series(r).rolling(20, min_periods=5).std() \
            .fillna(fallback).to_numpy(dtype=float)
    sig = np.maximum(np.asarray(sigma_hist, dtype=float), fallback * 0.2)
    theta = float(cfg.get("markov_kappa", 0.5)) * sig
    states = np.where(r > theta, 0, np.where(r < -theta, 2, 1)).astype(int)
    p = F.markov_independence_pvalue(states, 3)
    shrink = float(cfg.get("markov_shrink_weight", 0.5)) if p > float(cfg.get("markov_shrink_pvalue", 0.10)) else 0.0
    unconditional = np.bincount(states, minlength=3).astype(float)
    P = F.markov_transition_matrix(states, 3, shrink, unconditional)
    cur = int(states[-1])
    cur_name = "UFD"[cur]
    pi1 = np.zeros(3)
    pi1[cur] = 1.0
    m = MarkovState(matrix=P, current=cur_name, pvalue=p,
                    p_uu=float(P[0, 0]), p_dd=float(P[2, 2]),
                    exp_up=F.expected_streak(P[0, 0]), exp_dn=F.expected_streak(P[2, 2]))
    for h in horizons:
        pi = F.markov_horizon(pi1, P, h)
        m.pi_h[h] = pi
        m.p_markov_h[h] = float(F.clip_prob(pi[0], 0.02, 0.98))
    return m


# --------------------------------------------------------------------------- M7 蒙特卡洛 / 一致性

@dataclass
class SimResult:
    horizon: int
    p_up: float
    p_down: float
    prob_drop_5: float
    prob_drop_10: float
    prob_rise_5: float
    prob_rise_10: float
    q: np.ndarray
    mean: float
    var95: float
    var99: float
    cvar95: float
    mu_d: float
    sim_p_up: float
    method: str


def stable_seed(code: str, asof: date, seed_base: int) -> int:
    """seed = hash(code, asof_date, seed_base)，保证同一输入两次运行完全一致（§7.4 M7、§14 性质测试）。"""
    s = f"{code}|{asof.isoformat()}|{seed_base}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def _path_log_returns(z: np.ndarray, sigma_daily: np.ndarray, mu_d: float,
                      limit: float) -> np.ndarray:
    """逐日 r=μ_d+σ_k·z*，简单收益按涨跌停截断后累加（§4.4 / §7.4 M7）。"""
    daily = mu_d + sigma_daily[None, :] * z
    simple = F.clip_limit(np.expm1(daily), limit)
    return np.log1p(simple).sum(axis=1)


def simulate_fhs(vol: VolState, h: int, target_p_up: float, cfg: dict[str, Any],
                 z: np.ndarray, limit: float, drift_hint: float | None = None) -> SimResult:
    """FHS（过滤历史模拟）+ 联合一致性（§7.4 M7、§7.6 第 4 步）。

    入参 z：该股固定种子生成的 [n_paths, max_h] 标准化残差抽样，所有候选漂移共用，
    保证 P(R_h>0) 关于 μ_d 单调平滑。
    """
    n_paths = z.shape[0]
    sig = np.maximum(vol.sigma_daily[:h], 1e-8)
    zz = z[:, :h]
    mode = str(cfg.get("consistency_mode", "direction"))
    tol = 0.002
    if mode == "direction":
        # 未截断时的解析解作为二分起点：-h·μ = Q_{1-p}(Σ σ_k z_k)
        base = (sig[None, :] * zz).sum(axis=1)
        mu = float(-np.quantile(base, 1.0 - float(target_p_up)) / h)
        lo, hi = mu - 0.05 / h, mu + 0.05 / h
        best = mu
        for _ in range(int(cfg.get("consistency_iters", 24))):
            R = _path_log_returns(zz, sig, mu, limit)
            p = float((R > 0).mean())
            if abs(p - target_p_up) <= 0.0005:
                best = mu
                break
            if p < target_p_up:
                lo = mu
            else:
                hi = mu
            mu = 0.5 * (lo + hi)
            best = mu
    else:
        mu = float(drift_hint or 0.0) / h
        best = mu
    R = _path_log_returns(zz, sig, best, limit)
    # 保底求根：涨跌停截断会让 P(R>0) 关于 μ 的单调区间变窄，极端 p_up 时二分可能无法
    # 落到目标。对同一批随机数做位置平移（等价于再微调一次漂移 μ_d），可严格保证
    # P(R_h>0)=p_up，且不改变分布形状（§7.6「所有候选漂移共用同一批随机数」）。
    p_now = float((R > 0).mean())
    if abs(p_now - float(target_p_up)) > 0.0005:
        delta = -float(np.quantile(R, 1.0 - float(target_p_up)))
        R = R + delta
        best += delta / h
        p_now = float((R > 0).mean())
    simple = F.to_simple_return(R)
    sim_p_up = p_now
    t5 = float(cfg.get("drop_threshold_1", 0.05))
    t10 = float(cfg.get("drop_threshold_2", 0.10))
    r5 = float(cfg.get("rise_threshold_1", 0.05))
    r10 = float(cfg.get("rise_threshold_2", 0.10))
    q = np.quantile(simple, np.asarray(cfg["quantile_taus"], dtype=float))
    var95, cvar95 = F.var_cvar_from_samples(simple, 0.95)
    var99, _ = F.var_cvar_from_samples(simple, 0.99)
    return SimResult(
        horizon=int(h), p_up=float(target_p_up), p_down=float(1 - target_p_up),
        prob_drop_5=float((simple <= -t5).mean()), prob_drop_10=float((simple <= -t10).mean()),
        prob_rise_5=float((simple >= r5).mean()), prob_rise_10=float((simple >= r10).mean()),
        q=F.enforce_quantile_monotonic(q), mean=float(np.expm1(R.mean())),
        var95=max(var95, 0.0), var99=max(var99, 0.0), cvar95=max(cvar95, 0.0),
        mu_d=float(best), sim_p_up=sim_p_up, method="FHS",
    )


def analytic_t_distribution(mu_h: float, sigma_h: float, nu: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """location-scale t 解析分布（§7.7），walk-forward 的快速 OOS 用它代替 10000 条路径。"""
    scale = F.student_t_scale(max(nu, 2.5))
    sd = max(sigma_h * scale, 1e-8)
    taus = np.asarray(cfg["quantile_taus"], dtype=float)
    q_log = mu_h + sd * stats.t.ppf(taus, max(nu, 2.5))
    q = F.enforce_quantile_monotonic(F.to_simple_return(q_log))
    def p_below(x):
        return float(stats.t.cdf((math.log1p(x) - mu_h) / sd, max(nu, 2.5)))
    t5, t10 = float(cfg["drop_threshold_1"]), float(cfg["drop_threshold_2"])
    r5, r10 = float(cfg["rise_threshold_1"]), float(cfg["rise_threshold_2"])
    # 分位有解析解；CVaR 用 400 点下尾数值积分（CDF 概率 0..0.05 等距取样）
    df = max(nu, 2.5)
    var95 = -float(F.to_simple_return(mu_h + sd * stats.t.ppf(0.05, df)))
    var99 = -float(F.to_simple_return(mu_h + sd * stats.t.ppf(0.01, df)))
    grid = np.linspace(0.0002, 0.05, 400)
    r_q = F.to_simple_return(mu_h + sd * stats.t.ppf(grid, df))
    cvar95 = -float(r_q.mean())
    return dict(quantiles=q, p_up=float(stats.t.sf(-mu_h / sd, max(nu, 2.5))),
                prob_drop_5=p_below(-t5), prob_drop_10=p_below(-t10),
                prob_rise_5=1 - p_below(r5), prob_rise_10=1 - p_below(r10),
                mean=float(np.expm1(mu_h)), var95=max(var95, 0.0),
                var99=max(var99, 0.0), cvar95=max(cvar95, 0.0), method="location_scale_t")


# --------------------------------------------------------------------------- M9 EVT

@dataclass
class EvtFit:
    group: str
    u: float = 0.0
    xi: float = 0.0
    beta: float = 1.0
    n: int = 0
    n_u: int = 0
    var99: float = 0.0
    es99: float = 0.0
    ok: bool = False


def fit_evt(residuals: np.ndarray, group: str, cfg: dict[str, Any]) -> EvtFit:
    """下尾 POT/GPD：阈值 u=90% 分位，scipy.stats.genpareto 极大似然拟合（loc=0）（§7.5 M9）。"""
    a = np.asarray(residuals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 100:
        return EvtFit(group=group, n=int(a.size))
    loss = -a
    u = float(np.quantile(loss, float(cfg.get("evt_tail_quantile", 0.90))))
    exceed = loss[loss > u] - u
    if exceed.size < int(cfg.get("evt_min_exceedances", 50)):
        return EvtFit(group=group, u=u, n=int(a.size), n_u=int(exceed.size))
    try:
        xi, _, beta = stats.genpareto.fit(exceed, floc=0)
        xi = float(np.clip(xi, -0.5, 0.9))
        beta = float(max(beta, 1e-6))
        var, es = F.gpd_var_es(u, beta, xi, a.size, exceed.size, 0.99)
        return EvtFit(group=group, u=u, xi=xi, beta=beta, n=int(a.size),
                      n_u=int(exceed.size), var99=var, es99=es, ok=True)
    except Exception:
        return EvtFit(group=group, u=u, n=int(a.size), n_u=int(exceed.size))


# --------------------------------------------------------------------------- 历史描述指标

def historical_stats(r: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    """MDD / 年化 Sharpe / 年化波动（§7.5，仅历史描述，不参与预测）。"""
    a = np.asarray(r, dtype=float)
    a = a[np.isfinite(a)][-250:]
    if a.size < 20:
        return dict(mdd=0.0, sharpe=0.0, ann_vol=0.0)
    price = np.exp(np.cumsum(a))
    return dict(mdd=F.max_drawdown(price),
                sharpe=F.annualized_sharpe(a, float(cfg.get("rf", 0.02)),
                                           int(cfg.get("trading_days_per_year", 244))),
                ann_vol=float(np.std(a, ddof=1) * math.sqrt(cfg.get("trading_days_per_year", 244))))
