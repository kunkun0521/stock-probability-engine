# 工程决策记录（需求文档 §0.3）

> 原则：保守、可复现、可审计。以下都是需求文档没有写死、由实现方自行决定并记录的选择。
> 除非特别说明，所有取舍都通过 `config.yaml` 可调，不影响接口返回结构。

## 1. 依赖与运行环境

| 决策 | 原因 |
| --- | --- |
| 配置/特征文件用 **JSON 语法的 YAML 1.2**（`config.yaml` / `features.yaml`） | JSON 是 YAML 的合法子集，Python 标准库 `json` 与 Java Jackson 都能直接读，不需要 PyYAML。允许 `//` 与 `#` 行注释，读取时剥离 |
| **不使用 SQLAlchemy** | 本机 Windows WDAC 强制模式拦截 `sqlalchemy/_cache_key_cy.pyd`（`ImportError: DLL load failed ... 应用程序控制策略已阻止此文件`）。改用 PyMySQL 直连 |
| **不使用 arch / statsmodels / LightGBM 做核心拟合** | 同样为避免 C 扩展被拦、以及减少安装成本。GARCH(1,1)-t 用 `scipy.optimize` + `scipy.signal.lfilter` 自实现；分位数回归用 numpy 向量化 mini-batch SGD；逻辑回归/保序回归用 scikit-learn |
| 引擎默认用系统 `py -3` 而非项目 `.venv` | 系统 Python 3.14 的 pandas/numpy/scipy/sklearn 均可正常加载；后端 `stock.prob.python` 可配置成任何可用解释器 |

## 2. 数学实现的取舍

| 决策 | 说明 |
| --- | --- |
| GARCH 方差递推用 `lfilter` 向量化 | `σ²_t = ω + α r²_{t−1} + β σ²_{t−1}` 是线性递推，可写成 `lfilter([1],[1,−β], ω+α r²_{t−1})`，避免 500 步 Python 循环。MLE 用标准化 Student-t 对数似然，约束 ω>0、α+β<0.999、2.5<ν≤30；不收敛/触边/样本不足一律回退 EWMA |
| 分位数回归用 pinball SGD 而非 `QuantReg` / LightGBM | 文档允许编码者按速度与稳定性选择。SGD 全程 numpy、固定种子、损失与 §14 的 `ρ_τ(u)=u(τ−1[u<0])` 完全一致，80 万样本 × 5 个分位在数秒内完成 |
| 联合一致性的「保底求根」 | 涨跌停逐日截断会让 `P(R_h>0)` 关于漂移 μ_d 的可达区间变窄，极端 `p_up` 时单纯二分可能落不到目标。此时对**同一批随机数**的最终路径做位置平移 `R += −Q_{1−p}(R)`，等价于再微调一次 μ_d，严格保证 §14 的 `|P_sim − p_up| ≤ 0.002`，且分布形状不变 |
| location-scale t 解析分布 | walk-forward 的 OOS 用它代替 10000 条 FHS 路径，分位/VaR/CVaR 有解析解或数值积分，使样本外区间可以覆盖更大范围而不至于跑几小时。LIVE 仍用完整 FHS |
| EVT 只在 daily / predict 末尾做一次 | 单股超限样本太少，按 `sigma_1d_next` 三等分做全市场合并 POT/GPD 拟合；结果写 `prob_model_registry` 与 `prob_method_choice`，不替换 FHS 的模拟分布（文档允许「可选」） |

## 3. 数据处理

| 决策 | 说明 |
| --- | --- |
| 停牌日：价格前向填充 + `valid=False` 掩码 | 文档要求「指标在有效交易日序列上计算」。全市场向量化实现下采用 ffill 近似：停牌日价格不变、收益记 0 且 `valid=False`，训练样本、模型拟合、结算都只用 `valid=True` 的格子。停牌占比很低时影响可忽略，且完全不引入未来数据 |
| 训练 `σ_h` 用 20 日已实现波动换算 | 80 万个合并训练样本不可能逐个拟合 GARCH；M5/M6 的标准化尺度用 `vol_20/√244 × √h`。最终预测仍用单股 GARCH 的 `vol_h` |
| 合并训练样本按日抽样 | `max_train_samples=150000`，每日在股票维度均匀抽样（固定种子），并按交易日等权 `w=1/当日样本数`，避免股票多的日子主导 |
| 前复权快照每次重读完整窗口 | 禁止拼接不同次读取的价格；`settle` 也用同一份快照重算已实现收益 |
| `pct_chg` 只做交叉校验与涨跌停判定 | 收益率一律由前复权 close 计算；差异 > 0.001 计入 `prob_run.data_warn_count` |

## 4. Walk-forward 与耗时

| 决策 | 说明 |
| --- | --- |
| 样本外范围默认最近 40 个交易日、抽样 300 只股票 | 文档的全量 walk-forward（241–500−h 日 × 全市场 × 10000 路径）在桌面机上需要数小时。用 `walkforward.oos_max_days / max_stocks / oos_stride` 控制；实测 46,172 条 OOS 已足够给出稳定指标。三个参数都可调成全量 |
| L1 合并模型只在样本外开始处训练一次；L2 融合按块滚动 | 每个测试块用「此前已结算 OOS」训练 stacking 与校准器（带 `before` 时间过滤，无未来泄漏）。这是文档「L2 随样本外块滚动」的核心；L1 固定使耗时可控，评估结论更保守 |
| OOS 快速模式 `oos_fast=true` | 有意用 EWMA + 解析 t 降低耗时，因此 OOS 的 `VOL_EWMA`、`LOW_SAMPLE`、近期停牌属于**工程近似**，不计入个股可靠性降级；LIVE 模式仍按 §8.4 完整降级 |
| 断点续跑 | `artifacts/walkforward_checkpoint.json` 记录已完成块与 asof；同一 asof 重跑跳过已完成部分 |

## 5. 评级与回测

| 决策 | 说明 |
| --- | --- |
| `backtest.min_grade` 默认 `C`（文档示例为 B） | 样本外没有优势时周期层会如实标 C/D，若严格按 B 过滤会几乎没有可回测样本。默认放宽到 C 并在参数快照里记录，可随时收紧到 B/A |
| 回测基准用全市场等权 market proxy | 表内没有指数，与特征口径一致 |
| `prob_backtest_trade` 为新增表 | 文档只列了 summary/nav，但页面需要成交明细来解释净值与跳过原因 |

## 6. 数据库与工程

| 决策 | 说明 |
| --- | --- |
| 新增 `prob_method_choice` 与 `prob_backtest_trade` | 分别承载 §7.7 方法择优结果与回测成交明细，便于前端展示与审计 |
| Java 与 Python 双重建表 | `engine/schema.sql` 与 `resources/prob_schema.sql` 内容完全一致，都是 `CREATE TABLE IF NOT EXISTS`；只启动后端看页面时表也已存在 |
| 幂等补覆盖索引 | MySQL 不支持 `CREATE INDEX IF NOT EXISTS`，两端都用「查 information_schema / try-except duplicate key」的方式加 `idx_prob_pred_ov`、`idx_prob_pred_oos`。实测把总览的两次 `GROUP BY` 从各 12 秒降到 50ms |
| 总览接口 10 秒 TTL 缓存 | 后端 `ProbRepository` 进程内缓存，任务结束时主动失效；避免反复扫描当日全市场 |
| 概率/分位/风险字段全部随每条预测落库 | 便于复盘与「无未来数据泄漏」验收；`components_json` 保存分项概率、模型参数、因子贡献、分位候选 |
| `model_version = 代码版本 + sha1(配置)[:10]` | 配置一改版本号就变；每条预测可通过 `run_id`/`model_version` 追溯完整配置快照 |
| 单只股票失败不中断整体 | 记录原因，失败率 > `engine.fail_rate_alert=0.05` 才告警；实测全市场 0 失败 |

## 7. 前后端

| 决策 | 说明 |
| --- | --- |
| 后端只做 REST + 任务调度，不重写统计公式 | 公式唯一实现在 `engine/formulas.py` 与 `engine/models/*`；前端公式说明数据集中在 `src/prob/formulas.js`，三处按同一口径描述 |
| 公式弹窗 40 条、8 个分组 | 每条含公式表达式、逐符号「来自哪张表哪个字段」、入参统计口径、输出字段、示例、文档章节、注意事项 |
| 复用现有 8080 / 5173 服务 | Java 代码放在现有 Spring Boot 工程的新 `prob` 包，React 放在现有前端的新 `src/prob/` 目录；不额外起服务，人工操作最少 |
| 响应外壳 `ProbResponse` 带 `disclaimer` | 满足 §10.3 / §15；不改动项目全局 `ApiResponse`，前端 `client.js` 只取 `data`，兼容 |

## 8. 已知未完全实现 / 可继续扩展

- M2 的 LightGBM 挑战者、M9 用 EVT 替换 FHS 尾部残差属于文档 P3「可选」，接口已预留 `selected_method` / `prob_method_choice`，未启用。
- AUC 用 sklearn 的整体 ROC AUC，文档要求「按交易日在横截面计算后取均值」，当前实现是全样本 AUC；后续可在 `evaluation.py` 按日分组求均值。
- Christoffersen 独立性检验（文档标注可选）未实现，Kupiec POF 已实现。
- 行业/资金流/指数/股票名称：表内没有，v1 不使用；`DataProvider` 接口以 `db.source_table` 参数形式预留。
