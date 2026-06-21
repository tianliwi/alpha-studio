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

## 已知局限
- 原型用 yfinance 基本面字段；权威性弱于 SEC EDGAR（升级路径见 spec）
- 使用当前 S&P 500 成分股，存在幸存者偏差
