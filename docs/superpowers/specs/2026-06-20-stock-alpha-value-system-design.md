# 股票 Alpha 因子发掘系统 — 设计文档

**日期**: 2026-06-20
**状态**: 已批准，待实现
**作者**: brainstorming 协作产出

## 1. 目标与定位

构建一套**巴菲特风格的低频价值投资**股票 Alpha 因子发掘系统，面向**美股 S&P 500**，月度/季度调仓。系统覆盖数据采集、处理、因子计算、因子评估、ML 多因子合成、策略回测，并通过命令行输出排名股票清单与回测报告。

**核心理念**：可解释的基本面价值因子打底，ML 仅用于因子加权/组合优化，而非黑盒挖因子。先用免费数据源跑通原型，后续再考虑升级。

## 2. 范围与约束

- **市场**：美股，universe = S&P 500 当前成分股
- **频率**：低频，月度（默认）或季度调仓
- **数据源（原型）**：`yfinance`（价格 + 基本面字段）；后续升级路径为 SEC EDGAR companyfacts（更权威）
- **方法论**：基本面因子 + LightGBM 多因子合成打分
- **交互**：CLI（命令行/脚本），输出排名清单 + 回测报告
- **架构选型**：轻量模块化管线（best-of-breed 组合），非 Qlib 一体化框架

### 已知局限（原型阶段，文档化）
- **幸存者偏差**：原型使用当前 S&P 500 成分股，未用历史成分股。标注为已知局限，后续升级。
- **基本面数据质量**：yfinance 基本面字段不如官方财报权威，原型可接受，后续用 EDGAR 升级。

## 3. 架构与模块边界

6 个单一职责模块，通过 pandas DataFrame（`MultiIndex(date, ticker)`）+ parquet 文件通信，各自可独立测试。

```
alpha-studio/
├── data/          数据采集与缓存层
│   ├── prices         yfinance 拉日线 → parquet 缓存
│   ├── fundamentals   yfinance 基本面字段 → 标准化 → parquet（升级位：SEC EDGAR）
│   └── universe       维护 S&P 500 成分股列表
├── factors/       因子计算层（纯函数：财报+价格 → 因子值 DataFrame）
├── evaluation/    Alphalens 评估单因子有效性（IC/IR、分组、换手率）
├── model/         LightGBM 多因子合成打分（cross-sectional，月度）
├── backtest/      月频调仓回测引擎 + 交易成本，pyfolio 出报告
└── cli/           命令行入口，串联流程
```

### 接口约定
- 各层输入/输出均为带 `MultiIndex(date, ticker)` 的 pandas DataFrame，落地 parquet
- `data/` 层对上层屏蔽数据源差异——换付费 API 只改 `data/`，上层不动
- `factors/` 层每个因子是独立纯函数，便于单测与增删

### 设计原则
- 数据采集与因子计算解耦
- 因子定义与模型解耦（换因子不动模型，换模型不动因子）
- 模块小而聚焦，单一职责，接口清晰

## 4. 端到端数据流

```
1. universe   → 取当前 S&P 500 成分股列表
2. data       → 拉每只股票日线价格（open + close）+ 季度财报字段，缓存 parquet
3. factors    → 在每个调仓日 T 的收盘后计算因子横截面（估值因子用当日 close 派生 market_cap，每日刷新）
4. evaluation → (研究模式) Alphalens 看各因子 IC/IR，筛掉无效因子
5. model      → LightGBM 用历史因子→未来1月收益（open-to-open）训练，输出当期综合打分
6. backtest   → 按打分选 Top-N、月度调仓、T+1 开盘成交、扣成本，pyfolio 出报告
7. cli        → 打印当期排名清单 + 保存回测报告
```

### 执行机制（避免同日未来函数）
- **信号-成交时序**：调仓日 T **收盘后**用 T 的收盘价计算因子/打分 → 在 **T+1 交易日开盘价**成交建仓
- **因子计算口径**：估值类因子（PB、earnings yield、FCF yield、book-to-market）以 `market_cap = 当日 close × shares_out` 派生，随价格每日更新；纯财报类因子（ROE、ROA、毛利率、净利率、D/E）季度内不变
- **持有期收益口径**：与执行一致，按「本次成交开盘价 → 下次调仓成交开盘价」（open-to-open）计算，ML 标签与回测收益统一用此口径

## 5. 因子方法论

### 首批因子（巴菲特价值风格，可解释）

| 类别 | 因子 |
|------|------|
| 盈利能力 | ROE、ROA、毛利率、净利率 |
| 估值 | PE、PB、FCF Yield、EV/EBIT |
| 质量 | 负债率（D/E）、流动比率、盈利稳定性 |
| 成长 | 营收增速、EPS 增速（温和成长，非高估值成长股） |

### 防坑设计
- **Point-in-time / 滞后**：财报按发布日滞后使用（季报延后约 45–90 天），避免未来函数（look-ahead bias）
- **横截面标准化**：每个调仓日对因子做 z-score 或分位排名，再合成
- **缺失值处理**：财报缺失的股票当期剔除，不强行填充
- **幸存者偏差**：原型接受当前成分股偏差，文档标注

### ML 部分（务实，防过拟合）
- 模型：LightGBM 回归，目标 = 未来 1 个月（或 3 个月）收益
- 标签与特征严格时间对齐，**walk-forward 训练**（过去训、未来测），不做随机划分
- 防过拟合：限制树深、特征数少、定期重训

## 6. 回测引擎（月频，自建薄层）

- 调仓日 T 收盘算信号，**T+1 开盘价**成交；持有至下一调仓日的 T+1 开盘价
- 每次调仓按综合打分选 Top-N（如 Top 20–30 只），等权或按打分加权
- 持有期收益按 open-to-open 计算（与执行口径一致）
- 交易成本：每次换仓扣手续费+滑点（如单边 0.1%）
- 输出指标：年化收益、夏普、最大回撤、胜率、换手率、与 S&P 500 基准对比
- 用 `pyfolio` 生成 tearsheet 报告（图 + 表）

## 7. CLI 命令设计

```
sp run-pipeline                    # 全流程：拉数据→算因子→训练→回测→出报告
sp rank --date 2025-06             # 输出某月当期排名股票清单（实盘选股用）
sp fetch-data                      # 仅更新数据缓存
sp eval-factors                    # 仅跑 Alphalens 因子有效性诊断
sp backtest --start --end --topk   # 仅回测
```

## 8. 错误处理

- 数据层：单只股票拉取失败 → 记录日志跳过，不中断整体；网络重试
- 缓存：parquet 已存在则跳过重拉，支持增量更新
- 因子层：缺失数据当期剔除该股票并记录
- 全程结构化日志（loguru 或标准 logging）

## 9. 测试策略

- **因子函数**：小型构造数据单测，手算对比（如 ROE）
- **防未来函数**：专门测试财报滞后逻辑
- **回测引擎**：已知输入验证收益/调仓计算
- **数据层**：mock API 响应，不依赖真实网络
- 框架：pytest

## 10. 技术栈

- Python 3.11+，依赖管理 `uv` 或 `pip + requirements.txt`
- 核心库：pandas、numpy、yfinance、lightgbm、alphalens-reloaded、pyfolio-reloaded、pyarrow、loguru、typer(CLI)
- 测试：pytest

## 11. 可复用开源工程（调研结论）

- **Alphalens (alphalens-reloaded)**：因子有效性诊断标准工具，直接用于 evaluation 层
- **pyfolio (pyfolio-reloaded)**：组合风险收益分析与 tearsheet，用于回测报告
- **LightGBM**：因子合成 ML 模型
- **yfinance**：原型数据源
- **Microsoft Qlib**：评估后未采用为骨架（美股免费基本面接入摩擦大、框架重、黑盒多）；保留为未来参考

## 12. 未来升级路径

1. 基本面数据源 yfinance → SEC EDGAR companyfacts（point-in-time、更权威）
2. 引入历史成分股，消除幸存者偏差
3. 评估升级数据源至付费 API（如 Financial Modeling Prep）
4. 可选：扩展 universe 至中小盘或全美股
5. 可选：Web 仪表盘
