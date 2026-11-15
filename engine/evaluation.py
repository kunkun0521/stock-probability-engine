"""样本外评估、校准、方法择优与可靠性评级（文档 §8.2、§8.3、§8.4、§7.7）。

所有指标只用 status='SETTLED' 的 OOS 预测；显著性用按交易日 block bootstrap；
每项指标同时给出相对基准 B0（训练期基础上涨率）/B1（正态）/B2（随机游走 0.5）的差异。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import formulas as F
from .db import Db


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def load_oos(db: Db, cfg: dict[str, Any], before: date | None = None) -> pd.DataFrame:
    sql = ("SELECT code,market,asof_date,horizon,p_up,exp_ret,q05,q25,q50,q75,q95,"
           "vol_h,vol_1d_next,var95,cvar95,reliability_grade,components_json,"
           "selected_method,vol_model,realized_ret,realized_up,hit_dir,"
           "below_q05,above_q95,breach_var95 FROM prob_prediction "
           "WHERE pred_type='OOS' AND status='SETTLED' AND realized_ret IS NOT NULL")
    params: list[Any] = []
    if before:
        sql += " AND asof_date < %s"
        params.append(before)
    sql += " LIMIT 500000"
    return pd.DataFrame(db.query(sql, params))


def evaluate(db: Db, cfg: dict[str, Any], eval_date: date, logger=None) -> dict[str, Any]:
    df = load_oos(db, cfg)
    if df.empty:
        if logger:
            logger.warning("EVALUATE 无已结算的 OOS 样本，跳过")
        return dict(n=0, period_grade={})
    metric_rows: list[dict] = []
    calib_rows: list[dict] = []
    choice_rows: list[dict] = []
    period_grade: dict[int, str] = {}

    def add(h, scope, model, name, value, n=None, ci=(None, None), base=None):
        metric_rows.append(dict(eval_date=eval_date, horizon=int(h), scope=scope,
                                model_name=model, metric_name=name, value=float(value),
                                ci_low=ci[0], ci_high=ci[1], n_samples=int(n or 0),
                                base_value=base, delta=(float(value) - base) if base is not None else None,
                                created_at=None))

    for h, grp in df.groupby("horizon"):
        h = int(h)
        y = grp["realized_up"].to_numpy(dtype=float)
        p = grp["p_up"].to_numpy(dtype=float)
        comps = [_json(c) for c in grp["components_json"]]
        p0 = float(np.mean([c.get("p0", 0.5) for c in comps])) or 0.5
        n = len(y)
        # ---- 方向指标
        brier = F.brier_score(y, p)
        b0 = F.brier_score(y, np.full(n, p0))
        b2 = F.brier_score(y, np.full(n, 0.5))
        bss = F.brier_skill_score(y, p, p0)
        ll = F.logloss(y, p)
        try:
            auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
        except Exception:
            auc = 0.5
        ece = F.expected_calibration_error(y, p, int(cfg.get("calibration_bins", 10)))
        # BSS 的 block bootstrap CI：按交易日的日均 Brier 差值构造样本
        daily = []
        for _, g in grp.groupby("asof_date"):
            yy = g["realized_up"].to_numpy(dtype=float)
            pp = g["p_up"].to_numpy(dtype=float)
            daily.append(F.brier_score(yy, np.full(len(yy), p0)) - F.brier_score(yy, pp))
        ci = F.block_bootstrap_ci(np.asarray(daily, dtype=float), block=max(h, 1),
                                  n_boot=300, seed=int(cfg.get("seed_base", 20260101)) + h)
        add(h, "ALL", "M10", "Brier", brier, n, base=b0)
        add(h, "ALL", "B0", "Brier", b0, n)
        add(h, "ALL", "B1", "Brier", F.brier_score(
            y, np.clip(np.array([c.get("p_base", 0.5) for c in comps]), 0.01, 0.99)), n)
        add(h, "ALL", "B2", "Brier", b2, n)
        add(h, "ALL", "M10", "LogLoss", ll, n)
        add(h, "ALL", "M10", "AUC", auc, n, base=0.5)
        add(h, "ALL", "M10", "ECE", ece, n)
        add(h, "ALL", "M10", "BSS", bss, n, ci, base=0.0)
        # ---- 校准分箱
        bins = int(cfg.get("calibration_bins", 10))
        edges = np.linspace(0, 1, bins + 1)
        for b in range(bins):
            m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
            if not m.any():
                continue
            calib_rows.append(dict(eval_date=eval_date, horizon=h, model_name="M10",
                                   bin_no=b + 1, bin_low=float(edges[b]), bin_high=float(edges[b + 1]),
                                   n=int(m.sum()), mean_pred=float(p[m].mean()),
                                   freq_up=float(y[m].mean())))
        # ---- 幅度：MAE + 每日 Rank IC
        realized = grp["realized_ret"].to_numpy(dtype=float)
        exp = grp["exp_ret"].to_numpy(dtype=float)
        exp_cond = np.array([c.get("exp_ret_cond", np.nan) for c in comps], dtype=float)
        mae = float(np.mean(np.abs(exp - realized)))
        mae_cond = float(np.nanmean(np.abs(exp_cond - realized)))
        ics = [F.rank_ic(g["exp_ret"].to_numpy(float), g["realized_ret"].to_numpy(float))
               for _, g in grp.groupby("asof_date") if len(g) >= 5]
        ic_mean, ic_ir = float(np.mean(ics)), float(np.mean(ics) / max(np.std(ics), 1e-8)) if ics else 0.0
        add(h, "ALL", "M10", "MAE", mae, n, base=mae_cond)
        add(h, "ALL", "M5", "MAE", mae_cond, n)
        add(h, "ALL", "M10", "RankIC", ic_mean, len(ics))
        add(h, "ALL", "M10", "IC_IR", ic_ir, len(ics))
        # ---- 分位 / 区间覆盖率
        below = float(grp["below_q05"].fillna(0).mean())
        above = float(grp["above_q95"].fillna(0).mean())
        add(h, "ALL", "M10", "q05_coverage", below, n, base=0.05)
        add(h, "ALL", "M10", "q95_coverage", 1 - above, n, base=0.95)
        add(h, "ALL", "M10", "interval_coverage", 1 - below - above, n, base=0.90)
        # pinball loss：比较 location-scale t / 分位数回归（q_quantreg）
        taus = np.asarray(cfg["quantile_taus"], dtype=float)
        qs = grp[["q05", "q25", "q50", "q75", "q95"]].to_numpy(dtype=float)
        pin_t = float(np.mean([F.pinball_loss(realized - qs[:, j], taus[j])
                               for j in range(len(taus))]))
        q6 = np.array([c.get("q_quantreg") or [np.nan] * 5 for c in comps], dtype=float)
        pin_q = float(np.nanmean([F.pinball_loss(realized - q6[:, j], taus[j])
                                  for j in range(len(taus))])) if np.isfinite(q6).any() else 1e6
        add(h, "ALL", "location_scale_t", "pinball", pin_t, n)
        add(h, "ALL", "M6_quantile", "pinball", pin_q, n)
        choice_rows.append(dict(eval_date=eval_date, horizon=h, output_name="quantiles",
                                selected_method="M6_quantile" if pin_q < pin_t else "location_scale_t",
                                score=min(pin_q, pin_t),
                                candidates_json={"location_scale_t": pin_t, "M6_quantile": pin_q}))
        # ---- 风险：VaR 突破率 + Kupiec；波动 QLIKE
        breach = float(grp["breach_var95"].fillna(0).mean())
        kp = F.kupiec_pvalue(int(grp["breach_var95"].fillna(0).sum()), n, 0.95)
        add(h, "ALL", "M8", "var95_breach", breach, n, base=0.05)
        add(h, "ALL", "M8", "kupiec_pvalue", kp, n)
        rv = np.maximum(np.abs(realized), 1e-6) ** 2
        fv = np.maximum(grp["vol_h"].to_numpy(dtype=float), 1e-6) ** 2
        add(h, "ALL", "M4", "QLIKE", F.qlike(rv, fv), n)
        # ---- 方法择优：p_up 候选 M10 / M1 / M2；exp_ret 候选 分布均值 / M5
        cand = {}
        for name, key in (("M10", "p_up"), ("M1", "p_m1"), ("M2", "p_m2")):
            arr = grp[key].to_numpy(float) if key == "p_up" else \
                np.array([c.get(key, 0.5) for c in comps], dtype=float)
            cand[name] = F.brier_score(y, np.clip(arr, 0.01, 0.99))
        best = min(cand, key=cand.get)
        for name, v in cand.items():
            add(h, "ALL", name, "Brier", v, n)
        choice_rows.append(dict(eval_date=eval_date, horizon=h, output_name="p_up",
                                selected_method=best, score=cand[best],
                                candidates_json=cand))
        choice_rows.append(dict(eval_date=eval_date, horizon=h, output_name="exp_ret",
                                selected_method="M5" if mae_cond < mae else "distribution_mean",
                                score=min(mae, mae_cond),
                                candidates_json={"distribution_mean": mae, "M5": mae_cond}))
        # ---- 周期层可靠性评级
        grade = F.grade_from_period(bss, ci[0])
        period_grade[h] = grade
        add(h, "ALL", "M10", "reliability_grade_num", "ABCD".index(grade), n)
        # ---- 分月与波动分组
        for month, gm in grp.groupby(grp["asof_date"].map(lambda d: f"MONTH:{str(d)[:7]}")):
            add(h, str(month), "M10", "Brier", F.brier_score(
                gm["realized_up"].to_numpy(float), gm["p_up"].to_numpy(float)), len(gm))
        med = grp["vol_h"].median()
        for label, gm in (("BUCKET:low_vol", grp[grp["vol_h"] <= med]),
                          ("BUCKET:high_vol", grp[grp["vol_h"] > med])):
            if len(gm) >= 20:
                add(h, label, "M10", "Brier", F.brier_score(
                    gm["realized_up"].to_numpy(float), gm["p_up"].to_numpy(float)), len(gm))

    if metric_rows:
        cols = ["eval_date", "horizon", "scope", "model_name", "metric_name", "value",
                "ci_low", "ci_high", "n_samples", "base_value", "delta"]
        # prob_eval_metric 主键是自增 id，重跑覆盖的语义用「先删当日、再批量插入」实现
        db.execute("DELETE FROM prob_eval_metric WHERE eval_date=%s", (eval_date,))
        db.upsert("prob_eval_metric", metric_rows, cols, [])
    if calib_rows:
        db.execute("DELETE FROM prob_calibration_bin WHERE eval_date=%s", (eval_date,))
        db.upsert("prob_calibration_bin", calib_rows,
                  ["eval_date", "horizon", "model_name", "bin_no", "bin_low", "bin_high",
                   "n", "mean_pred", "freq_up"], [])
    if choice_rows:
        db.execute("DELETE FROM prob_method_choice WHERE eval_date=%s", (eval_date,))
        db.upsert("prob_method_choice", choice_rows,
                  ["eval_date", "horizon", "output_name", "selected_method", "score",
                   "candidates_json"], [])
    if logger:
        logger.info("EVALUATE n=%s horizons=%s grades=%s", len(df), sorted(period_grade), period_grade)
    return dict(n=int(len(df)), period_grade=period_grade,
                metrics=len(metric_rows), calibration=len(calib_rows))


def fit_evt_market(db: Db, cfg: dict[str, Any], eval_date: date,
                   pools: list[tuple[float, Any]], logger=None) -> dict[str, Any]:
    """M9 EVT：把全市场 GARCH 标准化残差按波动率三分组做合并 POT/GPD 拟合（§7.5 M9）。

    单股超限样本太少，必须合并；超限样本 < evt_min_exceedances 时回退 Student-t。
    结果写 prob_model_registry 与 prob_method_choice，供前端「方法择优」展示。
    """
    from dataclasses import asdict
    from .models import fit_evt
    if not pools:
        return {}
    sig = np.array([p[0] for p in pools], dtype=float)
    chunks = np.array_split(np.argsort(sig), 3)
    out: dict[str, Any] = {}
    choices = []
    for group, idx in zip(("low_vol", "mid_vol", "high_vol"), chunks):
        arr = np.concatenate([np.asarray(pools[int(i)][1], dtype=float) for i in idx])
        evt = fit_evt(arr, group, cfg)
        out[group] = asdict(evt)
        db.upsert("prob_model_registry", [dict(
            model_name=f"M9_evt_{group}", horizon=1, train_end_date=eval_date,
            artifact_path="", params_json=asdict(evt),
            metrics_json=dict(n=evt.n, n_u=evt.n_u, var99=evt.var99, es99=evt.es99))],
            ["model_name", "horizon", "train_end_date", "artifact_path", "params_json",
             "metrics_json"], ["artifact_path", "params_json", "metrics_json"])
        choices.append(dict(
            eval_date=eval_date, horizon=1, output_name=f"VaR_evt_{group}",
            selected_method="EVT" if evt.ok else "Student_t",
            score=evt.var99,
            candidates_json=dict(xi=evt.xi, beta=evt.beta, u=evt.u, n=evt.n,
                                 n_u=evt.n_u, var99=evt.var99, es99=evt.es99,
                                 fallback="Student_t" if not evt.ok else None)))
    # 用 LEFT() 而不是 LIKE 'VaR_evt_%'：pymysql 会把 LIKE 里的 % 当成参数占位符
    db.execute("DELETE FROM prob_method_choice WHERE eval_date=%s AND LEFT(output_name,8)='VaR_evt_'",
               (eval_date,))
    db.upsert("prob_method_choice", choices,
              ["eval_date", "horizon", "output_name", "selected_method", "score",
               "candidates_json"], [])
    if logger:
        logger.info("EVT %s", {k: dict(ok=v["ok"], xi=round(v["xi"], 3),
                                       var99=round(v["var99"], 3)) for k, v in out.items()})
    return out
