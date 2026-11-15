"""预测编排：训练合并模型 → 逐股单股模型 → M10 融合 → 分布/风险 → 落库行。

文档 §7.1 计算顺序：M4 波动 → M1/M2/M3 → M10 得到 p_up → M5 期望 → M6/M7 分布 → M8/M9 风险 → 评级。
LIVE 走完整 GARCH-t + FHS 蒙特卡洛；OOS（walk-forward）走快速模式（EWMA + location-scale t 解析分位），
以便在合理时间内覆盖全市场样本外区间，取舍见 docs/decisions.md。
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from . import formulas as F
from .config import BASE_DIR
from .data import MarketData
from .db import Db
from .features import FeatureStore, signal_columns
from .labels import LabelFrame, build_labels
from .models import (M2_FEATURES, ConditionalExpectation, DirectionLogit, FusionModel,
                     PinballQuantile, PooledModels, SignalBayes, TrainSamples,
                     analytic_t_distribution, build_train_samples, fit_markov,
                     fit_volatility, historical_stats, simulate_fhs, stable_seed)
from .store import ProbStore


def feature_names(fs: FeatureStore) -> list[str]:
    return [n for n in M2_FEATURES if n in fs.frames]


def train_pooled(md: MarketData, fs: FeatureStore, cfg: dict[str, Any], h: int,
                 positions: list[int], keys: list[str], train_end_pos: int,
                 train_end_date: date, store: ProbStore | None = None,
                 logger=None, fusion: FusionModel | None = None) -> PooledModels:
    """训练某周期的 M1/M2/M5/M6（+ 复用已训练的 M10），保存产物并登记。"""
    lf = build_labels(md, h, cfg)
    samples = build_train_samples(md, fs, lf, positions, keys, cfg,
                                  feature_names(fs), seed=int(cfg.get("seed_base", 20260101)) + h)
    pm = PooledModels(horizon=h, p0=samples.p0, train_end_pos=train_end_pos,
                      train_end_date=train_end_date, n_samples=int(samples.y.size))
    if samples.y.size < 200:
        if logger:
            logger.warning("horizon=%s 训练样本不足（%s 条），改用保守基线", h, samples.y.size)
        pm.m1 = SignalBayes(cfg, h, train_end_pos)
        pm.m2 = DirectionLogit(cfg, h)
        pm.m5 = ConditionalExpectation()
        pm.m6 = PinballQuantile(cfg, h)
        pm.m10 = fusion or FusionModel(cfg, h)
        return pm
    pm.m1 = SignalBayes(cfg, h, train_end_pos).fit(samples)
    pm.m2 = DirectionLogit(cfg, h).fit(samples)
    pm.pre = pm.m2.pre
    pm.m5 = ConditionalExpectation().fit(samples)
    pm.m6 = PinballQuantile(cfg, h).fit(samples)
    pm.m10 = fusion or FusionModel(cfg, h)
    pm.metrics = dict(
        p0=float(samples.p0), n_samples=int(samples.y.size), n_days=samples.n_days,
        mu_train=float(samples.r_h.mean()), sigma_h_mean=float(samples.sigma_h.mean()),
        k_up=pm.m5.k_up, k_dn=pm.m5.k_dn, best_c=pm.m2.best_c,
        cv_brier=pm.m2.cv_brier,
    )
    art_root = Path(cfg["engine"].get("artifacts_dir", "artifacts"))
    if not art_root.is_absolute():
        art_root = BASE_DIR / art_root
    for name, model in (("M1_signal", pm.m1), ("M2_logit", pm.m2),
                        ("M5_expectation", pm.m5), ("M6_quantile", pm.m6)):
        path = model.save(art_root / name / str(h) / train_end_date.isoformat() / "model.pkl")
        if store:
            store.save_registry(name, h, train_end_date, path, model.params(), pm.metrics)
    if store:
        store.save_signal_stats(pm.m1, h, train_end_date)
    if logger:
        logger.info("TRAIN h=%s samples=%s days=%s p0=%.4f k_up=%.3f k_dn=%.3f C=%s",
                    h, samples.y.size, samples.n_days, samples.p0,
                    pm.m5.k_up, pm.m5.k_dn, pm.m2.best_c)
    return pm


def train_fusion_from_oos(db: Db, cfg: dict[str, Any], h: int,
                          logger=None, before: date | None = None) -> FusionModel:
    """用样本外 L1 分项概率训练 stacking 元模型与校准器（§7.6、§8.1 L2）。

    before 不为空时只用该日期之前已结算的 OOS，实现滚动 L2，严禁未来数据泄漏。
    """
    fm = FusionModel(cfg, h)
    try:
        sql = ("SELECT components_json, realized_up FROM prob_prediction "
               "WHERE pred_type='OOS' AND horizon=%s AND status='SETTLED' "
               "AND realized_up IS NOT NULL AND components_json IS NOT NULL ")
        params: list[Any] = [h]
        if before:
            sql += "AND asof_date < %s "
            params.append(before)
        sql += "ORDER BY asof_date LIMIT 200000"
        rows = db.query(sql, params)
    except Exception:
        return fm
    comps, ys = [], []
    for r in rows:
        c = r["components_json"]
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except Exception:
                continue
        if isinstance(c, dict) and r["realized_up"] is not None:
            comps.append(c)
            ys.append(float(r["realized_up"]))
    if len(comps) >= 200:
        fm.fit(comps, np.asarray(ys, dtype=float))
        if logger:
            logger.info("FUSION h=%s n_oos=%s cal=%s ECE %.4f→%.4f w=%s",
                        h, fm.n_oos, fm.calibration_method, fm.ece_before, fm.ece_after,
                        np.round(fm.w, 3).tolist())
    return fm


def _downgrade(grade: str, n: int) -> str:
    order = "ABCD"
    return order[min(order.index(grade if grade in order else "C") + n, 3)]


def predict_cross_section(md: MarketData, fs: FeatureStore, pooled: dict[int, PooledModels],
                          cfg: dict[str, Any], pos: int, pred_type: str,
                          model_version: str, run_id: int | None,
                          *, fast: bool = False, keys: list[str] | None = None,
                          period_grade: dict[int, str] | None = None,
                          logger=None):
    """对某一交易日全股票池输出全部周期的预测行与单股模型状态行。"""
    keys = keys or md.universe
    horizons = list(cfg["horizons"])
    max_h = max(horizons)
    asof = md.dates[pos]
    ann = float(cfg.get("trading_days_per_year", 244))
    names = feature_names(fs)
    X = fs.cross_section(pos, names, keys).to_numpy(dtype=np.float32)
    sig = signal_columns(fs, pos, keys)
    active_lists = [[n for n, arr in sig.items() if arr[i]] for i in range(len(keys))]
    period_grade = period_grade or {}

    # 预计算与股票无关的合并模型输出
    p_m2 = {h: (pooled[h].m2.predict(X) if pooled[h].m2 else np.full(len(keys), 0.5))
            for h in horizons}
    # M6 分位数回归很便宜（纯矩阵乘法），OOS 快速模式也算，供方法择优比较 pinball loss
    q_m6 = {}
    for h in horizons:
        m6 = pooled[h].m6
        q_m6[h] = m6.predict(X, names) if m6 and m6.W.size else np.zeros((len(keys), 5))

    n_paths = int(cfg.get("n_paths", 10000)) if not fast else 0
    pred_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    evt_pools: list[tuple[float, np.ndarray]] = []
    failed = 0
    for i, key in enumerate(keys):
        try:
            code = md.code_of(key)
            market = md.market_of(key)
            series = md.arr["ret_1"].iloc[:pos + 1][key].to_numpy(dtype=np.float64)
            r = series[np.isfinite(series)]
            if r.size < 60:
                failed += 1
                continue
            vol = fit_volatility(r, cfg, max_h, force_ewma=fast)
            markov = fit_markov(r, cfg, horizons)
            limit = md.limit_of(key, asof)
            z = None
            if not fast:
                rng = np.random.default_rng(stable_seed(code, asof, int(cfg.get("seed_base", 20260101))))
                pool = vol.residual_pool
                if pool.size == 0:
                    pool = np.array([0.0], dtype=np.float32)
                z = pool[rng.integers(0, pool.size, size=(n_paths, max_h))]
            if vol.residual_pool.size:
                evt_pools.append((float(vol.sigma_1d_next), vol.residual_pool))
            hist = historical_stats(r, cfg)
            recent_susp = float(md.arr["suspended"].iloc[max(0, pos - 19):pos + 1][key].sum())
            recent_limit = float(fs.frames["limit_up_cnt_20"].iloc[pos].reindex([key]).iloc[0] or 0)
            contrib = pooled[horizons[0]].m2.contributions(X[i]) if pooled[horizons[0]].m2 else []

            for h in horizons:
                pm = pooled[h]
                sigma_h = float(np.sqrt(max(vol.var_by_h[h - 1], 1e-12)))
                mu_train = float(pm.metrics.get("mu_train", 0.0))
                p_base = F.normal_prob_up(mu_train, sigma_h)
                comp = dict(
                    p_m1=float(pm.m1.predict(active_lists[i])) if pm.m1 else 0.5,
                    p_m2=float(p_m2[h][i]),
                    p_markov=float(markov.p_markov_h.get(h, 1 / 3)),
                    p_base=float(p_base),
                    vol_pctile_self=float(vol.vol_pctile_self),
                )
                p_stack_raw = float(pm.m10.predict([{k: v for k, v in comp.items()}])[0]) \
                    if pm.m10 else float(p_m2[h][i])
                p_up = float(F.clip_prob(p_stack_raw, float(cfg.get("p_min", 0.05)),
                                          float(cfg.get("p_max", 0.95))))
                exp_cond = float(pm.m5.predict(p_up, sigma_h)) if pm.m5 else 0.0

                if fast:
                    df = max(vol.nu, 2.5)
                    sd = max(sigma_h * F.student_t_scale(df), 1e-8)
                    mu_h = float(-sd * stats.t.ppf(1.0 - p_up, df))
                    dist = analytic_t_distribution(mu_h, sigma_h, df, cfg)
                    sim_p_up = dist["p_up"]
                    mu_d = mu_h / h
                    dist_method = "location_scale_t"
                else:
                    sim = simulate_fhs(vol, h, p_up, cfg, z[:, :h], limit,
                                       drift_hint=math.log1p(max(exp_cond, -0.9)))
                    dist = dict(quantiles=sim.q, mean=sim.mean, var95=sim.var95,
                                var99=sim.var99, cvar95=sim.cvar95,
                                prob_drop_5=sim.prob_drop_5, prob_drop_10=sim.prob_drop_10,
                                prob_rise_5=sim.prob_rise_5, prob_rise_10=sim.prob_rise_10)
                    sim_p_up, mu_d, dist_method = sim.sim_p_up, sim.mu_d, sim.method
                q = np.asarray(dist["quantiles"], dtype=float)
                q6 = F.enforce_quantile_monotonic(
                    F.to_simple_return(np.asarray(q_m6.get(h, np.zeros((1, 5)))[i], dtype=float)
                                       * sigma_h)) if h in q_m6 else None

                flags = []
                if vol.model == "EWMA":
                    flags.append("VOL_EWMA")
                if exp_cond * dist["mean"] < 0 and abs(exp_cond) > 0.002:
                    flags.append("EXP_SIGN_MISMATCH")
                if abs(sim_p_up - p_up) > 0.002:
                    flags.append("CONSISTENCY_GAP")
                if vol.n_obs < int(cfg.get("grade_min_samples", 300)):
                    flags.append("LOW_SAMPLE")
                # OOS 快速模式是**有意**用 EWMA + 截断窗口来降低耗时，vol_model 回退与
                # 有效样本不足都属于工程近似，不算个股模型失败，因此不参与降级；
                # LIVE 模式仍按 §8.4 每命中一项降一档。
                down = (0 if fast else (int("VOL_EWMA" in flags) + int("LOW_SAMPLE" in flags))) \
                    + int("EXP_SIGN_MISMATCH" in flags) \
                    + (0 if fast else int(recent_susp > 0 or recent_limit >= 3))
                grade = _downgrade(period_grade.get(h, "C"), down)

                components = dict(
                    comp, p_stack_raw=p_stack_raw, exp_ret_cond=exp_cond,
                    vol_pctile_mkt=None, sim_p_up=sim_p_up, mu_d=mu_d,
                    sigma_h=sigma_h, nu=vol.nu, n_obs=vol.n_obs,
                    k_up=pm.m5.k_up if pm.m5 else None, k_dn=pm.m5.k_dn if pm.m5 else None,
                    mu_train=mu_train, p0=pm.p0,
                    markov_matrix=markov.matrix.tolist(), markov_pvalue=markov.pvalue,
                    markov_state=markov.current, p_uu=markov.p_uu, p_dd=markov.p_dd,
                    exp_up_streak=markov.exp_up, exp_dn_streak=markov.exp_dn,
                    contributions=contrib[:10],
                    q_quantreg=q6.tolist() if q6 is not None else None,
                    quantile_method=dist_method,
                )
                pred_rows.append(dict(
                    code=code, market=market, asof_date=asof, horizon=int(h),
                    pred_type=pred_type, model_version=model_version, run_id=run_id,
                    p_up=p_up, p_down=1.0 - p_up,
                    exp_ret=float(dist["mean"]), median_ret=float(q[2]),
                    q05=float(q[0]), q25=float(q[1]), q50=float(q[2]),
                    q75=float(q[3]), q95=float(q[4]),
                    vol_1d_next=float(vol.sigma_1d_next), vol_h=sigma_h,
                    prob_drop_5=float(dist["prob_drop_5"]), prob_drop_10=float(dist["prob_drop_10"]),
                    prob_rise_5=float(dist["prob_rise_5"]), prob_rise_10=float(dist["prob_rise_10"]),
                    var95=float(dist["var95"]), var99=float(dist["var99"]), cvar95=float(dist["cvar95"]),
                    risk_level=None, reliability_grade=grade,
                    components_json=components,
                    selected_method=(f"p_up=M10|exp_ret={dist_method}|quantiles={dist_method}"
                                     f"|vol={vol.model}|var={dist_method}"),
                    vol_model=vol.model, flags=",".join(flags), status="PENDING"))
            state_rows.append(dict(
                code=code, market=market, asof_date=asof, vol_model=vol.model,
                omega=vol.omega, alpha=vol.alpha, beta=vol.beta, nu=vol.nu,
                loglik=vol.loglik, converged=int(vol.converged), n_obs=vol.n_obs,
                lb_pvalue=vol.lb_pvalue, jb_pvalue=vol.jb_pvalue,
                sigma_1d_next=vol.sigma_1d_next, vol_pctile_self=vol.vol_pctile_self,
                vol_pctile_mkt=None, markov_state=markov.current,
                markov_matrix_json=markov.matrix.tolist(), markov_pvalue=markov.pvalue,
                mdd_250=hist["mdd"], sharpe_250=hist["sharpe"], ann_vol_250=hist["ann_vol"]))
        except Exception as e:
            failed += 1
            if logger and failed <= 5:
                logger.warning("PREDICT_FAIL %s %s: %s", key, type(e).__name__, e)

    # 横截面分位：风险等级取 vol_h 与 cvar95 分位较大者（§7.5）
    if pred_rows:
        import pandas as pd
        df = pd.DataFrame([(r["code"], r["market"], r["horizon"], r["vol_h"],
                            r["cvar95"], r["vol_1d_next"]) for r in pred_rows],
                          columns=["code", "market", "horizon", "vol_h", "cvar95", "v1"])
        for h in horizons:
            m = df["horizon"] == h
            pv = df.loc[m, "vol_h"].rank(pct=True)
            pc = df.loc[m, "cvar95"].rank(pct=True)
            pct = np.maximum(pv.to_numpy(), pc.to_numpy())
            idx = np.flatnonzero(m.to_numpy())
            for j, row_i in enumerate(idx):
                pred_rows[row_i]["risk_level"] = F.risk_level_from_pctile(
                    float(pct[j]), float(cfg.get("risk_low_cut", 0.30)),
                    float(cfg.get("risk_mid_cut", 0.70)), float(cfg.get("risk_high_cut", 0.90)))
                pred_rows[row_i]["components_json"]["vol_pctile_mkt"] = float(
                    df.loc[m, "v1"].rank(pct=True).to_numpy()[j])
        for s in state_rows:
            key = f"{s['code']}.{s['market']}"
            row = df[df["code"] == s["code"]]
            if not row.empty:
                s["vol_pctile_mkt"] = float(row["v1"].rank(pct=True).iloc[0]) if len(df) > 1 else 0.5
    return pred_rows, state_rows, dict(n_universe=len(keys), n_ok=len(pred_rows),
                                        n_fail=failed, warn_count=md.warn_count,
                                        excluded=md.excluded, completeness=md.completeness,
                                        evt_pools=evt_pools)
