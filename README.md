# alpha-studio

巴菲特风格价值投资 alpha 因子发掘系统（S&P 500，月度调仓）。

## 安装
    python -m pip install -e .

## 用法
    sp fetch-data                       # 更新数据缓存
    sp eval-factors                     # 因子有效性诊断（Alphalens）
    sp backtest --start 2018-01-01 --end 2024-12-31 --topk 25
    sp rank --date 2024-06              # 当期排名股票清单
    sp run-pipeline                     # 全流程

## 数据源
- **基本面**：默认 SEC EDGAR companyfacts（全历史，免费，无 key）。可在 `config.FUNDAMENTALS_SOURCE`
  切换为 `"yfinance"`（仅近 ~2 年）。EDGAR 把 us-gaap XBRL 标签归一化为标准字段，
  流量取单季（3 个月）值并由"全年−前三季"派生 Q4。
- **价格**：yfinance 日线（开盘+收盘，复权）。

## 已知局限
- 使用当前 S&P 500 成分股，存在幸存者偏差
- EDGAR 总负债/自由现金流由组合概念派生（长期+流动债务之和；OCF−Capex），
  个别公司标签缺失时该字段为 NaN（横截面标准化可容忍）
