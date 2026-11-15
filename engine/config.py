"""统一配置加载（文档 §0.1、§12.2）。

config.yaml / features.yaml 使用 **JSON 语法的 YAML 1.2**（JSON 是 YAML 的合法子集）：
这样 Python 用标准库 json 即可零依赖读取，Java 端用 Jackson 也能直接 readTree，
不需要引入 PyYAML（本机 WDAC 策略会拦截部分 C 扩展 .pyd）。
文件里允许 // 与 # 行注释，加载时统一剔除，方便人工修改时写备注。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BASE_DIR / "config.yaml"
DEFAULT_FEATURES = BASE_DIR / "features.yaml"


def _strip_comments(text: str) -> str:
    """去掉整行注释与行尾 // 注释；字符串里的 //（如 URL）由 http:// 特判保护。"""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        # 去掉行尾 // 注释，但保留 http:// 之类协议串
        pos = line.find("//")
        if pos >= 0 and not line[max(0, pos - 1):pos + 2].endswith(":/"):
            if line[:pos].count('"') % 2 == 0:
                line = line[:pos]
        out.append(line)
    return "\n".join(out)


def read_json_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(_strip_comments(p.read_text(encoding="utf-8")))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or os.environ.get("PROB_CONFIG") or DEFAULT_CONFIG)
    cfg = read_json_yaml(p)
    cfg["__config_path__"] = str(p)
    # 环境变量覆盖数据库连接，方便 Docker / CI / 后端调度时注入
    if os.environ.get("PROB_DB_URL"):
        cfg["db"]["url"] = os.environ["PROB_DB_URL"]
    if os.environ.get("PROB_MAX_STOCKS"):
        cfg["engine"]["max_stocks"] = int(os.environ["PROB_MAX_STOCKS"])
    if os.environ.get("PROB_N_PATHS"):
        cfg["n_paths"] = int(os.environ["PROB_N_PATHS"])
    return cfg


def load_features(path: str | Path | None = None) -> dict[str, Any]:
    return read_json_yaml(path or DEFAULT_FEATURES)


def model_version(cfg: dict[str, Any]) -> str:
    """代码版本 + 配置哈希，写入 prob_run.model_version / prob_prediction.model_version（§10.2 可复现）。"""
    payload = json.dumps({k: v for k, v in cfg.items() if not k.startswith("__")},
                         sort_keys=True, ensure_ascii=False, default=str)
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{cfg.get('engine_code_version', 'spe')}-{h}"


def config_snapshot(cfg: dict[str, Any]) -> str:
    """写进 prob_run.config_json 的完整配置快照。"""
    return json.dumps({k: v for k, v in cfg.items() if not k.startswith("__")},
                      ensure_ascii=False, default=str)
