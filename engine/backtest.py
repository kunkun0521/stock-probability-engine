"""简单回测：检验样本外预测是否可交易（文档 §8.5）。

t+1 开盘买入，t+1+h 收盘卖出（满足 T+1，比标签口径更保守）；
买入日一字涨停视为买不到，卖出日跌停顺延；成本含佣金、印花税、滑点；
组合等权、每日新开仓、持仓相互重叠（每笔占 1/h 资金）。
注意：样本期仅约两年且行情阶段单一，结果页必须标注这一点。
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from . import formulas as F
from .data import MarketData
from .db import Db


def run_backtest(md: MarketData, db: Db, cfg: dict[str, Any], run_id: int | None,
                 logger=None) -> int | None:
    bt = cfg.get("backtest", {})
    h = int(bt.get("horizon", min(cfg["horizons"])))
    top_n = int(cfg.get("top_n", 20))
    min_grade = str(bt.get("min_grade", "B"))
    sort_by = str(bt.get("sort_by", "p_up"))
    rows = db.query(
        "SELECT code,market,asof_date,p_up,exp_ret,vol_h,reliability_grade "
        "FROM prob_prediction WHERE pred_type='OOS' AND horizon=%s "
        "AND status='SETTLED' ORDER BY asof_date", (h,))
    if not rows:
        if logger:
            logger.warning("BACKTEST 无已结算 OOS 预测，跳过")
        return None
    df = pd.DataFrame(rows)
    grade_order = {g: i for i, g in enumerate("ABCD")}
    df = df[df["reliability_grade"].map(lambda g: grade_order.get(str(g), 3))
            <= grade_order.get(min_grade, 1)]
    if sort_by == "exp_ret/vol_h":
        df["score"] = df["exp_ret"] / df["vol_h"].replace(0, np.nan)
        sort_col = "score"
    elif sort_by in df.columns:
        sort_col = sort_by
    else:
        sort_col = "p_up"
    dates = md.dates
    pos = {d: i for i, d in enumerate(dates)}
    o, c, hi, lo, pre = (md.arr[x] for x in ("open", "close", "high", "low", "preclose"))
    eps = float(cfg.get("limit_epsilon", 0.001))
    trades: list[dict] = []
    daily_ret: dict[int, list[float]] = {}
    for asof, grp in df.groupby("asof_date"):
        p0 = pos.get(asof)
        if p0 is None or p0 + 1 + h >= len(dates):
            continue
        buy_d, sell_d = dates[p0 + 1], dates[p0 + 1 + h]
        picks = grp.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False).head(top_n)
        for _, row in picks.iterrows():
            key = f"{row['code']}.{row['market']}"
            if key not in md.arr["open"].columns:
                continue
            buy = float(o.at[buy_d, key]) if np.isfinite(o.at[buy_d, key]) else 0.0
            lim = md.limit_of(key, buy_d)
            pre_buy = float(pre.at[buy_d, key]) if np.isfinite(pre.at[buy_d, key]) else 0.0
            one_word = (pre_buy > 0 and buy >= pre_buy * (1 + lim) - eps
                        and abs(float(hi.at[buy_d, key]) - float(lo.at[buy_d, key])) < 1e-6)
            if buy <= 0 or one_word:
                trades.append(dict(code=row["code"], market=row["market"], signal_date=asof,
                                   buy_date=buy_d, p_up=float(row["p_up"]),
                                   skipped="ONE_WORD_LIMIT_UP" if one_word else "NO_OPEN_PRICE"))
                continue
            # 卖出日跌停则顺延到下一个可卖交易日
            sd = sell_d
            for _ in range(5):
                if not md.is_limit_down(key, sd):
                    break
                nxt = pos.get(sd, -1) + 1
                if nxt >= len(dates):
                    sd = None
                    break
                sd = dates[nxt]
            if sd is None:
                trades.append(dict(code=row["code"], market=row["market"], signal_date=asof,
                                   buy_date=buy_d, p_up=float(row["p_up"]),
                                   skipped="LIMIT_DOWN_NO_EXIT"))
                continue
            sell = float(c.at[sd, key]) if np.isfinite(c.at[sd, key]) else 0.0
            if sell <= 0:
                continue
            net = F.backtest_trade_return(buy, sell, float(cfg.get("commission", 0.00025)),
                                          float(cfg.get("stamp_duty", 0.0005)),
                                          float(cfg.get("slippage", 0.001)))
            trades.append(dict(code=row["code"], market=row["market"], signal_date=asof,
                               buy_date=buy_d, sell_date=sd, p_up=float(row["p_up"]),
                               buy_price=buy, sell_price=sell, net_return=net, skipped=None))
            daily_ret.setdefault(pos[sd], []).append(net)

    if not daily_ret:
        if logger:
            logger.warning("BACKTEST 没有可成交的交易")
        return None
    all_days = sorted(set(list(daily_ret.keys()) +
                          [pos[d] for d in dates if pos[d] >= min(daily_ret)]))
    nav, bench, dd = [1.0], [1.0], [0.0]
    peak = 1.0
    mr = md.market_ret
    for i in all_days:
        r = float(np.mean(daily_ret[i])) / h if i in daily_ret else 0.0
        br = float(mr.iloc[i]) if i < len(mr) else 0.0
        nav.append(nav[-1] * (1 + r))
        bench.append(bench[-1] * (1 + br))
        peak = max(peak, nav[-1])
        dd.append((peak - nav[-1]) / peak)
    n_av, n_bv = np.asarray(nav), np.asarray(bench)
    met = F.portfolio_metrics(n_av, int(cfg.get("trading_days_per_year", 244)),
                              float(cfg.get("rf", 0.02)))
    bench_met = F.portfolio_metrics(n_bv, int(cfg.get("trading_days_per_year", 244)),
                                    float(cfg.get("rf", 0.02)))
    wins = [t["net_return"] for t in trades if t.get("net_return") is not None]
    win_rate = float(np.mean([w > 0 for w in wins])) if wins else 0.0
    win_p = F.binomial_winrate_pvalue(sum(1 for w in wins if w > 0), len(wins), 0.5)
    summary = dict(
        run_id=run_id, params_json=dict(top_n=top_n, horizon=h, min_grade=min_grade,
                                        sort_by=sort_by, commission=cfg.get("commission"),
                                        stamp_duty=cfg.get("stamp_duty"),
                                        slippage=cfg.get("slippage")),
        horizon=h, start_date=dates[min(daily_ret)], end_date=dates[max(daily_ret)],
        annual_return=met["annualReturn"], annual_vol=met["annualVol"],
        sharpe=met["sharpe"], max_drawdown=met["maxDrawdown"], win_rate=win_rate,
        win_pvalue=win_p, turnover=float(len(wins) / max(len(all_days), 1) / top_n),
        n_trades=len(wins), bench_annual_return=bench_met["annualReturn"],
        total_return=float(n_av[-1] - 1), bench_total_return=float(n_bv[-1] - 1))
    bid = db.insert("prob_backtest_summary", summary)
    nav_rows = [dict(backtest_id=bid, trade_date=dates[i] if i < len(dates) else dates[-1],
                     nav=float(n_av[j]), bench_nav=float(n_bv[j]), drawdown=float(dd[j]))
                for j, i in enumerate([min(all_days) - 1] + all_days)]
    db.upsert("prob_backtest_nav", nav_rows,
              ["backtest_id", "trade_date", "nav", "bench_nav", "drawdown"],
              ["nav", "bench_nav", "drawdown"])
    if trades:
        db.upsert("prob_backtest_trade", [dict(backtest_id=bid, **t) for t in trades],
                  ["backtest_id", "code", "market", "signal_date", "buy_date", "sell_date",
                   "p_up", "buy_price", "sell_price", "net_return", "skipped"], [])
    if logger:
        logger.info("BACKTEST id=%s trades=%s ann=%.2f%% bench=%.2f%% sharpe=%.2f mdd=%.2f%%",
                    bid, len(wins), met["annualReturn"] * 100,
                    bench_met["annualReturn"] * 100, met["sharpe"], met["maxDrawdown"] * 100)
    return int(bid)
