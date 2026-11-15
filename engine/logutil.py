"""结构化日志：每个任务记录耗时、成功/失败数量与失败原因（文档 §12.3）。"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


def get_logger(cfg: dict[str, Any], name: str = "prob") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(str(cfg.get("engine", {}).get("log_level", "INFO")).upper())
    log_dir = Path(cfg.get("engine", {}).get("log_dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent.parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / f"engine-{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


@contextmanager
def task_timer(logger: logging.Logger, task: str, **context):
    t0 = time.perf_counter()
    logger.info("TASK_START %s %s", task, json.dumps(context, ensure_ascii=False, default=str))
    state: dict[str, Any] = dict(ok=0, fail=0, reasons=[])
    try:
        yield state
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        logger.exception("TASK_FAIL %s %.2fs %s", task, time.perf_counter() - t0, e)
        raise
    finally:
        logger.info("TASK_END %s %.2fs %s", task, time.perf_counter() - t0,
                    json.dumps(state, ensure_ascii=False, default=str)[:2000])
