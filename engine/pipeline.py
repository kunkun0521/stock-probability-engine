"""任务编排：prepare / train / predict / walkforward / settle / evaluate / backtest / daily。

自动化目标（文档 §10.1、§13 P3）：`engine daily` 一条命令跑完全链路，
按 retrain_every 自动决定是否需要重训与滚动评估，支持按 asof_date 断点续跑。
任何一步都被包进 prob_run，记录耗时、成功/失败数量与配置快照。
"""
from __future__ import annotations

import json
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .backtest import run_backtest
from .config import BASE_DIR, config_snapshot, load_config, load_features, model_version
from .data import MarketData
from .db import Db
from .evaluation import evaluate, fit_evt_market
from .features import build_features
from .labels import training_positions, walkforward_blocks
from .logutil import get_logger, task_timer
from .models import PooledModels
from .predictor import predict_cross_section, train_fusion_from_oos, train_pooled
from .store import ProbStore


class Pipeline:
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or load_config()
        self.db = Db(self.cfg)
        self.store = ProbStore(self.db, self.cfg)
        self.logger = get_logger(self.cfg)
        self.version = model_version(self.cfg)
        self.db.ensure_schema()

    # ------------------------------------------------------------------ 基础
    def prepare(self, asof: date | None = None, max_stocks: int = 0):
        md = MarketData(self.cfg, self.db).load(asof=asof, max_stocks=max_stocks)
        fs = build_features(md, self.cfg)
        if md.completeness_message:
            self.logger.warning("DATA_WARN %s", md.completeness_message)
        self.logger.info("PREPARE asof=%s dates=%s stocks=%s universe=%s excluded=%s warn=%s",
                         md.asof, len(md.dates), len(md.keys), len(md.universe),
                         md.excluded, md.warn_count)
        return md, fs

    def _start(self, run_type: str, asof: date | None, run_id: int | None = None,
               max_stocks: int = 0):
        return self.store.start_run(run_type, asof, self.version, run_id)

    def period_grades(self) -> dict[int, str]:
        try:
            rows = self.db.query(
                "SELECT horizon,value FROM prob_eval_metric "
                "WHERE metric_name='reliability_grade_num' AND scope='ALL' "
                "AND eval_date=(SELECT MAX(eval_date) FROM prob_eval_metric)")
            return {int(r["horizon"]): "ABCD"[max(0, min(3, int(float(r["value"]))))]
                    for r in rows}
        except Exception:
            return {}

    def _keys_for_oos(self, md: MarketData) -> list[str]:
        n = int(self.cfg["walkforward"].get("max_stocks", 0) or 0)
        if n <= 0 or n >= len(md.universe):
            return md.universe
        rng = np.random.default_rng(int(self.cfg.get("seed_base", 20260101)))
        idx = np.sort(rng.choice(len(md.universe), size=n, replace=False))
        return [md.universe[i] for i in idx]

    def _train_all(self, md: MarketData, fs, positions_by_h: dict[int, list[int]],
                   keys: list[str], fusions: dict[int, Any] | None = None):
        pooled: dict[int, PooledModels] = {}
        for h in self.cfg["horizons"]:
            pos_list = positions_by_h[h]
            end_pos = int(pos_list[-1]) if pos_list else len(md.dates) - 1
            pooled[h] = train_pooled(md, fs, self.cfg, h, pos_list, keys, end_pos,
                                     md.dates[end_pos], self.store, self.logger,
                                     fusion=(fusions or {}).get(h))
        return pooled

    def settle_all(self, max_batches: int = 30) -> int:
        total = 0
        for _ in range(max_batches):
            n = self.store.settle(self.version, self.cfg.get("__settle_to") or date.today())
            total += n
            if n == 0:
                break
        if total:
            self.logger.info("SETTLE rows=%s", total)
        return total

    # ------------------------------------------------------------------ 各命令
    def cmd_prepare(self, asof=None, max_stocks=0, run_id=None):
        t = self.cfg.get("__asof_to") or date.today()
        rid = self._start("PREPARE", None, run_id, max_stocks)
        try:
            md, _ = self.prepare(asof, max_stocks)
            self.store.finish_run(rid, "SUCCESS", n_universe=len(md.universe),
                                  n_ok=len(md.universe), excluded=md.excluded,
                                  warn_count=md.warn_count,
                                  message=f"asof={md.asof} 完整性={md.completeness:.1%}")
            return dict(runId=rid, asof=str(md.asof), nUniverse=len(md.universe),
                        excluded=md.excluded, dataWarnCount=md.warn_count,
                        completeness=md.completeness)
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=str(e))
            raise

    def cmd_train(self, asof=None, max_stocks=0, run_id=None):
        rid = self._start("TRAIN", None, run_id, max_stocks)
        try:
            md, fs = self.prepare(asof, max_stocks)
            fusions = {h: train_fusion_from_oos(self.db, self.cfg, h, self.logger)
                       for h in self.cfg["horizons"]}
            positions = {h: training_positions(self.cfg, len(md.dates), h)
                         for h in self.cfg["horizons"]}
            pooled = self._train_all(md, fs, positions, md.universe, fusions)
            self.store.finish_run(rid, "SUCCESS", n_universe=len(md.universe),
                                  n_ok=len(md.universe), warn_count=md.warn_count,
                                  message="合并模型训练完成：" +
                                          ", ".join(f"h{h}={pooled[h].n_samples}" for h in pooled))
            return {"runId": rid, "samples": {h: pooled[h].n_samples for h in pooled}}
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=str(e))
            raise

    def cmd_predict(self, asof=None, max_stocks=0, run_id=None, fast=False,
                    pred_type="LIVE"):
        rid = self._start("PREDICT", None, run_id, max_stocks)
        try:
            md, fs = self.prepare(asof, max_stocks)
            self.cfg["__settle_to"] = md.asof
            fusions = {h: train_fusion_from_oos(self.db, self.cfg, h, self.logger)
                       for h in self.cfg["horizons"]}
            positions = {h: training_positions(self.cfg, len(md.dates), h)
                         for h in self.cfg["horizons"]}
            pooled = self._train_all(md, fs, positions, md.universe, fusions)
            rows, states, stats = predict_cross_section(
                md, fs, pooled, self.cfg, md.date_pos(md.asof), pred_type,
                self.version, rid, fast=fast, period_grade=self.period_grades(),
                logger=self.logger)
            self.store.save_predictions(rows)
            self.store.save_model_states(states)
            fit_evt_market(self.db, self.cfg, date.today(), stats.get("evt_pools", []), self.logger)
            rate = stats["n_fail"] / max(stats["n_universe"], 1)
            if rate > float(self.cfg["engine"].get("fail_rate_alert", 0.05)):
                self.logger.warning("FAIL_RATE %.2f%% 超过阈值", rate * 100)
            self.store.finish_run(rid, "SUCCESS", n_universe=stats["n_universe"],
                                  n_ok=len(rows), n_fail=stats["n_fail"],
                                  excluded=stats["excluded"], warn_count=stats["warn_count"],
                                  message=f"asof={md.asof} 预测 {len(rows)} 条，失败率 {rate:.2%}")
            return dict(runId=rid, asof=str(md.asof), rows=len(rows),
                        states=len(states), failed=stats["n_fail"])
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=str(e))
            raise

    def cmd_walkforward(self, asof=None, max_stocks=0, run_id=None, resume=True):
        rid = self._start("WALKFORWARD", None, run_id, max_stocks)
        try:
            md, fs = self.prepare(asof, max_stocks)
            self.cfg["__settle_to"] = md.asof
            blocks = walkforward_blocks(md, self.cfg)
            keys = self._keys_for_oos(md)
            ckpt = BASE_DIR / str(self.cfg["walkforward"].get(
                "checkpoint_file", "artifacts/walkforward_checkpoint.json"))
            done = 0
            if resume and ckpt.exists():
                try:
                    state = json.loads(ckpt.read_text(encoding="utf-8"))
                    if state.get("asof") == str(md.asof) and state.get("keys") == len(keys):
                        done = int(state.get("done", 0))
                except Exception:
                    done = 0
            h_max = max(self.cfg["horizons"])
            init_end = max(int(self.cfg.get("warmup_days", 60)), blocks[0]["test_start"] - h_max - 1)
            init_positions = {h: list(range(int(self.cfg.get("warmup_days", 60)),
                                            max(int(self.cfg.get("warmup_days", 60)) + 1,
                                                init_end - h + 1)))
                              for h in self.cfg["horizons"]}
            pooled = self._train_all(md, fs, init_positions, keys)
            total = 0
            for bi, block in enumerate(blocks):
                if bi < done:
                    continue
                # 滚动 L2：只用当前测试块之前已结算的 OOS 训练融合与校准
                fusions = {}
                for h in self.cfg["horizons"]:
                    fusions[h] = train_fusion_from_oos(self.db, self.cfg, h, self.logger,
                                                       before=md.dates[block["test_start"]])
                    pooled[h].m10 = fusions[h]
                for pos in range(block["test_start"], block["test_end"] + 1):
                    rows, states, stats = predict_cross_section(
                        md, fs, pooled, self.cfg, pos, "OOS", self.version, rid,
                        fast=bool(self.cfg.get("oos_fast", True)), keys=keys,
                        period_grade=self.period_grades(), logger=self.logger)
                    self.store.save_predictions(rows)
                    total += len(rows)
                self.settle_all()
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpt.write_text(json.dumps(dict(asof=str(md.asof), keys=len(keys),
                                                done=bi + 1, updated=datetime.now().isoformat()),
                                           ensure_ascii=False), encoding="utf-8")
                self.logger.info("WALKFORWARD block %s/%s done rows=%s",
                                 bi + 1, len(blocks), total)
            self.store.finish_run(rid, "SUCCESS", n_universe=len(keys), n_ok=total,
                                  excluded=md.excluded, warn_count=md.warn_count,
                                  message=f"{len(blocks)} 个样本外块，累计 {total} 条 OOS 预测")
            return dict(runId=rid, blocks=len(blocks), rows=total, keys=len(keys))
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=f"{e}\n{traceback.format_exc()[-800:]}")
            raise

    def cmd_settle(self, run_id=None):
        rid = self._start("SETTLE", None, run_id)
        try:
            n = self.settle_all()
            self.store.finish_run(rid, "SUCCESS", n_ok=n, message=f"结算 {n} 条")
            return dict(runId=rid, rows=n)
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=str(e))
            raise

    def cmd_evaluate(self, run_id=None):
        rid = self._start("EVAL", None, run_id)
        try:
            result = evaluate(self.db, self.cfg, date.today(), self.logger)
            self.store.finish_run(rid, "SUCCESS", n_ok=result.get("n", 0),
                                  message=json.dumps(result, ensure_ascii=False))
            return dict(runId=rid, **result)
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=str(e))
            raise

    def cmd_backtest(self, run_id=None, max_stocks=0):
        rid = self._start("BACKTEST", None, run_id)
        try:
            md, _ = self.prepare(None, max_stocks)
            bid = run_backtest(md, self.db, self.cfg, rid, self.logger)
            self.store.finish_run(rid, "SUCCESS", n_ok=int(bid is not None),
                                  message=f"backtest_id={bid}")
            return dict(runId=rid, backtestId=bid)
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=str(e))
            raise

    # ------------------------------------------------------------------ 一键日常
    def cmd_daily(self, asof=None, max_stocks=0, run_id=None, force=False):
        """prepare →（按间隔）walkforward/train → LIVE 预测 → settle → evaluate → backtest。"""
        rid = self._start("PREDICT", None, run_id, max_stocks)
        steps: list[str] = []
        try:
            with task_timer(self.logger, "daily", asof=str(asof), version=self.version) as t:
                md, fs = self.prepare(asof, max_stocks)
                self.cfg["__settle_to"] = md.asof
                steps.append("prepare")

                # 是否需要（重新）生成样本外：没有 OOS 或距上次超过 retrain_every 个交易日
                n_oos = self.db.scalar(
                    "SELECT COUNT(*) FROM prob_prediction WHERE pred_type='OOS'") or 0
                last_wf = self.db.scalar(
                    "SELECT MAX(asof_date) FROM prob_run WHERE run_type='WALKFORWARD' "
                    "AND status='SUCCESS'")
                gap = 999
                if last_wf:
                    try:
                        later = [d for d in md.dates if d > last_wf]
                        gap = len(later)
                    except Exception:
                        gap = 999
                need_wf = force or n_oos < 1000 or gap >= int(self.cfg.get("retrain_every", 20))
                if need_wf:
                    # 重置 checkpoint，让本次按当前 asof 增量跑（已完成的块会跳过）
                    self.cmd_walkforward(asof, max_stocks, None, resume=True)
                    steps.append("walkforward")
                else:
                    steps.append("walkforward(skip)")
                t["ok"] += 1

                self.settle_all()
                eval_result = evaluate(self.db, self.cfg, date.today(), self.logger)
                grades = self.period_grades()
                steps.append("evaluate")
                t["ok"] += 1

                fusions = {h: train_fusion_from_oos(self.db, self.cfg, h, self.logger)
                           for h in self.cfg["horizons"]}
                positions = {h: training_positions(self.cfg, len(md.dates), h)
                             for h in self.cfg["horizons"]}
                pooled = self._train_all(md, fs, positions, md.universe, fusions)
                steps.append("train")

                rows, states, stats = predict_cross_section(
                    md, fs, pooled, self.cfg, md.date_pos(md.asof), "LIVE",
                    self.version, rid, fast=False, period_grade=grades, logger=self.logger)
                self.store.save_predictions(rows)
                self.store.save_model_states(states)
                steps.append(f"predict:{len(rows)}")
                t["ok"] += len(rows)
                t["fail"] += stats["n_fail"]

                settled = self.settle_all()
                steps.append(f"settle:{settled}")
                evaluate(self.db, self.cfg, date.today(), self.logger)
                evt = fit_evt_market(self.db, self.cfg, date.today(),
                                     stats.get("evt_pools", []), self.logger)
                steps.append(f"evt:{len(evt)}")
                bid = run_backtest(md, self.db, self.cfg, rid, self.logger)
                steps.append(f"backtest:{bid}")
                self.store.finish_run(
                    rid, "SUCCESS", n_universe=stats["n_universe"], n_ok=len(rows),
                    n_fail=stats["n_fail"], excluded=stats["excluded"],
                    warn_count=stats["warn_count"],
                    message=" | ".join(steps) + f" | OOS={eval_result.get('n', 0)}")
            return dict(runId=rid, asof=str(md.asof), steps=steps, rows=len(rows),
                        oos=eval_result.get("n", 0), backtestId=bid)
        except Exception as e:
            self.store.finish_run(rid, "FAILED", message=f"{e}\n{traceback.format_exc()[-800:]}")
            self.logger.exception("DAILY FAILED")
            raise
