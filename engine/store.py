"""prob_* 表的读写封装（文档 §9、§10.2）。

所有写入都是 INSERT ... ON DUPLICATE KEY UPDATE（幂等）；同一主键重跑覆盖。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from .config import config_snapshot
from .db import Db

PRED_COLS = [
    "code", "market", "asof_date", "horizon", "pred_type", "model_version", "run_id",
    "p_up", "p_down", "exp_ret", "median_ret", "q05", "q25", "q50", "q75", "q95",
    "vol_1d_next", "vol_h", "prob_drop_5", "prob_drop_10", "prob_rise_5", "prob_rise_10",
    "var95", "var99", "cvar95", "risk_level", "reliability_grade", "components_json",
    "selected_method", "vol_model", "flags", "status", "settle_date", "realized_ret",
    "realized_up", "hit_dir", "below_q05", "above_q95", "breach_var95",
]
PRED_UPDATE = [c for c in PRED_COLS if c not in
               ("code", "market", "asof_date", "horizon", "pred_type", "model_version")]

STATE_COLS = [
    "code", "market", "asof_date", "vol_model", "omega", "alpha", "beta", "nu",
    "loglik", "converged", "n_obs", "lb_pvalue", "jb_pvalue", "sigma_1d_next",
    "vol_pctile_self", "vol_pctile_mkt", "markov_state", "markov_matrix_json",
    "markov_pvalue", "mdd_250", "sharpe_250", "ann_vol_250",
]
STATE_UPDATE = [c for c in STATE_COLS if c not in ("code", "market", "asof_date")]


class ProbStore:
    def __init__(self, db: Db, cfg: dict[str, Any]):
        self.db, self.cfg = db, cfg

    # ------------------------------------------------------------ prob_run
    def start_run(self, run_type: str, asof: date, model_version: str,
                  run_id: int | None = None) -> int:
        if run_id:
            self.db.update("prob_run", dict(status="RUNNING", asof_date=asof,
                                            started_at=datetime.now(), message=None,
                                            finished_at=None), "run_id=%s", (run_id,))
            return int(run_id)
        return self.db.insert("prob_run", dict(
            run_type=run_type, asof_date=asof, config_json=config_snapshot(self.cfg),
            model_version=model_version, status="RUNNING", n_universe=0, n_ok=0, n_fail=0,
            excluded_json={}, data_warn_count=0))

    def finish_run(self, run_id: int, status: str, *, n_universe=0, n_ok=0, n_fail=0,
                   excluded=None, warn_count=0, message=""):
        self.db.update("prob_run", dict(
            status=status, n_universe=n_universe, n_ok=n_ok, n_fail=n_fail,
            excluded_json=excluded or {}, data_warn_count=warn_count,
            finished_at=datetime.now(), message=message[:2000]), "run_id=%s", (run_id,))

    def run(self, run_id: int) -> dict[str, Any] | None:
        rows = self.db.query("SELECT * FROM prob_run WHERE run_id=%s", (run_id,))
        return rows[0] if rows else None

    # ------------------------------------------------------------ 预测 / 模型状态
    def save_predictions(self, rows: list[dict[str, Any]]) -> int:
        return self.db.upsert("prob_prediction", rows, PRED_COLS, PRED_UPDATE)

    def save_model_states(self, rows: list[dict[str, Any]]) -> int:
        return self.db.upsert("prob_stock_model_state", rows, STATE_COLS, STATE_UPDATE)

    def save_signal_stats(self, m1, h: int, train_end: date) -> int:
        rows = [dict(signal_key=k, horizon=h, train_end_date=train_end,
                     n=int(v["n"]), k=int(v["k"]), n_eff=v["n_eff"], p_hat=v["p_hat"],
                     posterior_mean=v["posterior_mean"], ci_low=v["ci_low"],
                     ci_high=v["ci_high"], p0=m1.p0, prior_strength=m1.m,
                     half_life=int(m1.half_life))
                for k, v in m1.stat.items()]
        if not rows:
            return 0
        return self.db.upsert("prob_signal_stat", rows,
                              ["signal_key", "horizon", "train_end_date", "n", "k", "n_eff",
                               "p_hat", "posterior_mean", "ci_low", "ci_high", "p0",
                               "prior_strength", "half_life"],
                              ["n", "k", "n_eff", "p_hat", "posterior_mean", "ci_low",
                               "ci_high", "p0", "prior_strength", "half_life"])

    def save_registry(self, model_name: str, h: int, train_end: date,
                      artifact_path: str, params: dict, metrics: dict) -> int:
        return self.db.upsert("prob_model_registry", [dict(
            model_name=model_name, horizon=h, train_end_date=train_end,
            artifact_path=artifact_path, params_json=params, metrics_json=metrics)],
            ["model_name", "horizon", "train_end_date", "artifact_path", "params_json", "metrics_json"],
            ["artifact_path", "params_json", "metrics_json"])

    # ------------------------------------------------------------ 结算
    def settle(self, model_version: str, asof_to: date, limit: int = 20000) -> int:
        """对已到期的 PENDING 预测补 realized 字段（§10.1 settle）。

        用同一张 prob_prediction 的自连接：找 t+h 对应的交易日收盘价无法只靠本表得到，
        因此这里用「按窗口交易日序号」的 SQL 关联 stock_daily 前复权收盘价。
        """
        table = self.cfg["db"].get("source_table", "stock_daily")
        flag = int(self.cfg["db"].get("adjustflag", 2))
        pending = self.db.query(
            "SELECT code,market,asof_date,horizon,pred_type,model_version,q05,q95,var95,p_up "
            "FROM prob_prediction WHERE status='PENDING' AND asof_date<=%s "
            "LIMIT %s", (asof_to, int(limit)))
        if not pending:
            return 0
        # 交易日历一次取回，避免逐行查库
        dates = [r["trade_date"] for r in self.db.query(
            f"SELECT DISTINCT trade_date FROM `{table}` WHERE adjustflag=%s "
            f"ORDER BY trade_date DESC LIMIT %s", (flag, int(self.cfg["lookback_days"])))]
        dates = sorted(dates)
        pos = {d: i for i, d in enumerate(dates)}
        by_code: dict[tuple[str, str], list[dict]] = {}
        for r in pending:
            by_code.setdefault((r["code"], r["market"]), []).append(r)
        out = []
        for (code, market), rows in by_code.items():
            need = set()
            for r in rows:
                p0 = pos.get(r["asof_date"])
                if p0 is None or p0 + int(r["horizon"]) >= len(dates):
                    continue
                need.add(r["asof_date"])
                need.add(dates[p0 + int(r["horizon"])])
            if not need:
                continue
            need = sorted(need)
            ph = ",".join(["%s"] * len(need))
            prices = {x["trade_date"]: float(x["close"]) for x in self.db.query(
                f"SELECT trade_date,close FROM `{table}` WHERE adjustflag=%s AND code=%s "
                f"AND market=%s AND trade_date IN ({ph}) AND close>0",
                (flag, code, market, *need))}
            for r in rows:
                p0 = pos.get(r["asof_date"])
                if p0 is None or p0 + int(r["horizon"]) >= len(dates):
                    continue
                settle_date = dates[p0 + int(r["horizon"])]
                c0, c1 = prices.get(r["asof_date"]), prices.get(settle_date)
                if not c0 or not c1:
                    continue
                realized = float(c1 / c0 - 1.0)
                up = int(realized > 0)
                hit = int((float(r["p_up"]) > 0.5) == bool(up))
                out.append(dict(
                    code=code, market=market, asof_date=r["asof_date"],
                    horizon=int(r["horizon"]), pred_type=r["pred_type"],
                    model_version=r["model_version"],
                    status="SETTLED", settle_date=settle_date, realized_ret=realized,
                    realized_up=up, hit_dir=hit,
                    below_q05=int(realized < float(r["q05"] or -9)),
                    above_q95=int(realized > float(r["q95"] or 9)),
                    breach_var95=int(realized < -abs(float(r["var95"] or 0))),
                    run_id=None, p_up=None, p_down=None, exp_ret=None, median_ret=None,
                    q05=None, q25=None, q50=None, q75=None, q95=None,
                    vol_1d_next=None, vol_h=None, prob_drop_5=None, prob_drop_10=None,
                    prob_rise_5=None, prob_rise_10=None, var95=None, var99=None, cvar95=None,
                    risk_level=None, reliability_grade=None, components_json=None,
                    selected_method=None, vol_model=None, flags=None))
        if not out:
            return 0
        # 只更新结算字段，不碰预测字段（避免覆盖 None）
        cols = ["code", "market", "asof_date", "horizon", "pred_type", "model_version",
                "status", "settle_date", "realized_ret", "realized_up", "hit_dir",
                "below_q05", "above_q95", "breach_var95"]
        upd = [c for c in cols if c not in
               ("code", "market", "asof_date", "horizon", "pred_type", "model_version")]
        return self.db.upsert("prob_prediction", out, cols, upd)

    def _close_at(self, code: str, market: str, d: date, table: str, flag: int) -> float | None:
        v = self.db.scalar(
            f"SELECT close FROM `{table}` WHERE adjustflag=%s AND code=%s "
            f"AND market=%s AND trade_date=%s", (flag, code, market, d))
        return float(v) if v else None
