"""命令行入口（文档 §10.1）：python -m engine.cli <command> [options]

    engine prepare                      读取窗口、完整性检查、确定股票池
    engine train                        训练合并模型 M1/M2/M5/M6
    engine predict --asof latest        生成 LIVE 全市场预测
    engine walkforward                  生成 OOS 样本外预测（滚动 L2、断点续跑）
    engine settle                       结算已到期预测
    engine evaluate                     计算指标、校准、方法择优、可靠性评级
    engine backtest                     按 §8.5 回测
    engine daily                        一键：prepare→walkforward→train→predict→settle→evaluate→backtest
    engine status                       查看当前数据状态与最近任务
    engine config                       输出当前生效配置（JSON）

所有命令输出一行 JSON，便于后端调度器解析。--asof 支持 YYYY-MM-DD 或 latest。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

from .config import BASE_DIR, load_config, model_version


def _date(s: str | None):
    if not s or s == "latest":
        return None
    return date.fromisoformat(s)


def _emit(obj):
    print(json.dumps(obj, ensure_ascii=False, default=str))


def cmd_status(pipe) -> dict:
    db = pipe.db
    tables = ["prob_run", "prob_prediction", "prob_stock_model_state", "prob_signal_stat",
              "prob_model_registry", "prob_eval_metric", "prob_calibration_bin",
              "prob_method_choice", "prob_backtest_summary", "prob_backtest_nav"]
    counts = {}
    for t in tables:
        try:
            counts[t] = int(db.scalar(f"SELECT COUNT(*) FROM `{t}`") or 0)
        except Exception:
            counts[t] = -1
    latest = {}
    try:
        for k, sql in (
            ("latestAsOf", "SELECT MAX(asof_date) FROM prob_prediction"),
            ("latestLive", "SELECT MAX(asof_date) FROM prob_prediction WHERE pred_type='LIVE'"),
            ("settled", "SELECT COUNT(*) FROM prob_prediction WHERE status='SETTLED'"),
            ("oosSettled", "SELECT COUNT(*) FROM prob_prediction WHERE pred_type='OOS' AND status='SETTLED'"),
        ):
            latest[k] = db.scalar(sql)
    except Exception:
        pass
    runs = []
    try:
        runs = db.query("SELECT run_id,run_type,asof_date,status,n_ok,n_fail,started_at,"
                        "finished_at,message FROM prob_run ORDER BY run_id DESC LIMIT 10")
    except Exception:
        pass
    return dict(code=0, modelVersion=pipe.version, tables=counts, data=latest, runs=runs,
                disclaimer=pipe.cfg.get("disclaimer", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="engine", description="A股股票概率预测引擎")
    ap.add_argument("--config", default=None, help="配置文件路径（默认项目根 config.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--asof", default="latest")
        p.add_argument("--limit-stocks", type=int, default=0, help="只处理前 N 只（调试用，0=全市场）")
        p.add_argument("--run-id", type=int, default=None, help="复用后端已创建的 prob_run 记录")
        p.add_argument("--fast", action="store_true", help="快速模式（EWMA + 解析 t，不做蒙特卡洛）")
        p.add_argument("--force", action="store_true", help="daily：强制重跑滚动评估")
        p.add_argument("--no-resume", action="store_true", help="walkforward：忽略断点从头跑")

    for name in ("prepare", "train", "predict", "walkforward", "settle", "evaluate",
                 "backtest", "daily", "status", "config"):
        common(sub.add_parser(name))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.cmd == "config":
        _emit(dict(code=0, data={k: v for k, v in cfg.items() if not k.startswith("__")},
                   modelVersion=model_version(cfg)))
        return 0
    try:
        from .pipeline import Pipeline
        pipe = Pipeline(cfg)
        asof = _date(args.asof)
        cmd = args.cmd
        if cmd == "status":
            _emit(cmd_status(pipe))
        elif cmd == "prepare":
            _emit(dict(code=0, data=pipe.cmd_prepare(asof, args.limit_stocks, args.run_id)))
        elif cmd == "train":
            _emit(dict(code=0, data=pipe.cmd_train(asof, args.limit_stocks, args.run_id)))
        elif cmd == "predict":
            _emit(dict(code=0, data=pipe.cmd_predict(asof, args.limit_stocks, args.run_id,
                                                     args.fast)))
        elif cmd == "walkforward":
            _emit(dict(code=0, data=pipe.cmd_walkforward(asof, args.limit_stocks,
                                                         args.run_id, not args.no_resume)))
        elif cmd == "settle":
            _emit(dict(code=0, data=pipe.cmd_settle(args.run_id)))
        elif cmd == "evaluate":
            _emit(dict(code=0, data=pipe.cmd_evaluate(args.run_id)))
        elif cmd == "backtest":
            _emit(dict(code=0, data=pipe.cmd_backtest(args.run_id, args.limit_stocks)))
        elif cmd == "daily":
            _emit(dict(code=0, data=pipe.cmd_daily(asof, args.limit_stocks, args.run_id,
                                                   args.force)))
        return 0
    except Exception as e:
        _emit(dict(code=1, message=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:]))
        return 1


if __name__ == "__main__":
    sys.exit(main())
