# A股股票概率预测引擎 · Python 计算引擎

> 统计计算全部在这里；Spring Boot 只负责 REST 与任务调度。所有输出都是统计估计，不构成投资建议。

## 快速开始

```bash
cd D:/code/PythonProject/stock-probability-engine

# 依赖（系统 Python 3.11+；本机 WDAC 环境下推荐 py -3）
py -3 -m pip install -r requirements.txt

# 一键：数据准备 → 滚动评估 → 训练 → 全市场预测 → 结算 → 评估 → EVT → 回测
py -3 -m engine.cli daily

# 查看当前数据状态
py -3 -m engine.cli status

# 单元测试（需求文档第 14 章基准 + 性质测试）
py -3 -m unittest discover -s tests -v
```

完整使用说明见 `D:/code/doc/股票概率预测引擎-使用说明.md`。

## 命令一览（§10.1）

| 命令 | 内容 |
| --- | --- |
| `engine prepare` | 读取 500 日窗口、完整性检查、确定股票池，写 `prob_run` |
| `engine train` | 训练合并模型 M1/M2/M5/M6（每个周期一套），产物落 `artifacts/` 并登记 |
| `engine predict --asof latest` | 单股模型 → 方向/幅度/风险 → 融合校准 → 落库（LIVE） |
| `engine walkforward` | 生成 OOS 样本外预测、滚动训练 L2 融合与校准、断点续跑 |
| `engine settle` | 对到期预测补 `realized_*` / `hit_dir` / 区间与 VaR 覆盖标记 |
| `engine evaluate` | 指标、校准分箱、方法择优、可靠性评级 |
| `engine backtest` | §8.5 简单回测 |
| `engine daily` | 以上按间隔自动编排，一条命令跑完 |
| `engine status` | 各表行数、最新基准日、最近 10 条运行记录 |
| `engine config` | 输出当前生效配置 JSON（供后端/脚本读取） |

通用参数：`--config <path>`、`--asof YYYY-MM-DD|latest`、`--limit-stocks N`（调试）、
`--run-id N`（复用后端创建的 `prob_run` 记录）、`--fast`（EWMA + 解析 t，不做蒙特卡洛）、
`--force`（daily 强制重跑滚动评估）、`--no-resume`（walk-forward 忽略断点）。

环境变量：`PROB_DB_URL`、`PROB_CONFIG`、`PROB_MAX_STOCKS`、`PROB_N_PATHS`。

## 目录

```
engine/
  config.py       配置加载、model_version（代码版本 + 配置哈希）
  db.py           PyMySQL 直连、幂等建表、批量 UPSERT、覆盖索引
  schema.sql      prob_* 全部表（与 Java resources/prob_schema.sql 一致）
  logutil.py      结构化日志、任务耗时
  data.py         读取/清洗/股票池/涨跌停/market proxy
  features.py     50+ 特征 + 12 个离散信号（形状 [日期 × 股票]）
  labels.py       R_h / Y_h、purge、walk-forward 分块
  formulas.py     ★ 全部数学公式的唯一实现（纯函数，单元测试直接调用）
  models/
    base.py       fit / predict / save / load 统一接口
    pooled.py     M1 贝叶斯信号、M2 逻辑回归、M5 期望、M6 分位数回归、M10 融合校准
    single.py     M3 马尔可夫、M4 GARCH-t/EWMA、M7 FHS 蒙特卡洛、M8 VaR/CVaR、M9 EVT
  predictor.py    M1–M10 编排：横截面预测 + 单股模型状态 + 风险评级
  store.py        prob_* 读写封装、settle
  evaluation.py   样本外指标、校准、方法择优、EVT 合并拟合
  backtest.py     成交、成本、净值、绩效
  pipeline.py     daily 全链路编排、断点续跑、失败容错
  cli.py          命令行入口（输出一行 JSON）
config.yaml       全部可配置参数
features.yaml     特征注册表（公式 / 回看 / 入参来源 / 统计口径）
tests/            单元测试
docs/             使用说明、公式说明、模型说明、decisions.md
artifacts/        模型产物、walk-forward checkpoint
logs/             结构化日志
```

## 配置要点

`config.yaml` 是 **JSON 语法的 YAML 1.2**（JSON 是 YAML 子集），允许 `//` / `#` 行注释。
改配置会改变 `model_version`，不会与旧预测混淆。关键项：

- 数据：`lookback_days=500`、`horizons=[1,3,5,10]`、`universe_prefixes`、`min_history=250`、`limit_pct`
- 方向：`prior_strength=20`、`half_life=60`、`min_n=30`、`l2_c_grid`
- 波动：`garch_refit_every`、`ewma_lambda=0.94`、`garch_min_obs=250`
- 分布：`n_paths=10000`、`consistency_mode=direction`、`p_min/p_max=0.05/0.95`
- 样本外：`walkforward.oos_max_days=40`、`max_stocks=300`、`oos_stride=5`、`retrain_every=20`
- 回测：`top_n=20`、`commission`、`stamp_duty`、`slippage`、`backtest.min_grade=C`

## 数据表

新增 11 张 `prob_` 表，不修改 `stock_daily`。核心是 `prob_prediction`
（主键 `code,market,asof_date,horizon,pred_type,model_version`），
每次预测的分项概率、模型参数、因子贡献都在 `components_json` 里，便于复盘审计。

## 自动化与可复现

- `daily` 按 `retrain_every` 自动决定是否重跑滚动评估；已完成的 OOS 块由 checkpoint 跳过。
- 所有写入都是 `INSERT ... ON DUPLICATE KEY UPDATE`，同一主键重跑覆盖。
- 单只股票失败不中断整体；失败率 > 5% 告警。
- 随机种子 `hash(code, asof_date, seed_base)`；`prob_run.config_json` 保存完整配置快照。
- `asof` 之后的行情根本不读进内存，天然满足「无未来数据泄漏」验收。

## 已验证

- 17 个单元测试全过（含第 14 章全部基准、一致性 ≤0.002、可复现性）。
- 2026-11-15 全市场 `daily` 端到端：4 分 32 秒、0 失败，
  12,284 条 LIVE 预测、46,172 条 OOS、GARCH 收敛率约 92%、EVT 三组拟合成功、799 笔回测。

## 注意

本机 Windows WDAC 会拦截部分 C 扩展 `.pyd`（SQLAlchemy、pandas 3.1.x 等）。
引擎刻意不依赖这些包；若 `.venv` import 失败，用系统 `py -3`。
详见 `docs/decisions.md`。
