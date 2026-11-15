"""模型统一接口 fit / predict / save / load（文档 §12.3）。

新增模型只要继承 BaseModel 并实现 fit/predict，调用方（pipeline / predictor）无需改动。
产物落在 artifacts/{model_name}/{horizon}/{train_end_date}/，数据库 prob_model_registry 只登记路径与参数。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


class BaseModel:
    name = "base"
    version = "1.0"

    def fit(self, *args, **kwargs) -> "BaseModel":
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def params(self) -> dict[str, Any]:
        """写进 prob_model_registry.params_json 的可读参数（系数、k_up/k_dn 等）。"""
        return {}

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta = p.with_suffix(".json")
        meta.write_text(json.dumps(dict(name=self.name, version=self.version,
                                       params=self.params()), ensure_ascii=False,
                                  indent=2, default=str), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
