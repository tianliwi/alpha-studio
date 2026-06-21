"""临时脚本：策略累计净值 vs SPY，使用相同的 open-to-open 成交时点。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from alpha_studio.config import TOP_N, TRANSACTION_COST
from alpha_studio.data.universe import get_sp500_tickers
from alpha_studio.data.prices import fetch_prices, execution_open_prices
from alpha_studio.cli.main import _build_scores, _rebalance_dates, _snap_to_trading_days
from alpha_studio.backtest.engine import select_topn, run_backtest

START, END = "2021-06-20", "2026-06-20"

tickers = get_sp500_tickers()
scores, exec_prices = _build_scores(tickers, START, END)
weights = select_topn(scores, TOP_N)
result = run_backtest(weights, exec_prices, TRANSACTION_COST)
strat = result["returns"]  # index = 信号日 T，值=该期 open-to-open 收益

# 重建调仓日全集，找每个持有期的结束边界
daily = fetch_prices(tickers, START, END)
rebal = _snap_to_trading_days(_rebalance_dates(START, END), daily)

# 用同样的 execution_open_prices 逻辑构造 SPY 的 T+1 开盘成交价
spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
spy_open = spy_raw["Open"]
spy_open.columns = ["SPY"]
spy_daily = spy_open.stack(future_stack=True).rename("open").to_frame()
spy_daily.index = spy_daily.index.set_names(["date", "ticker"])
spy_exec = execution_open_prices(spy_daily, rebal)
spy_op = spy_exec["exec_open"].unstack("ticker").sort_index()["SPY"]
spy_ret = spy_op.pct_change().shift(-1)  # 本期成交开盘 -> 下期成交开盘
spy_ret = spy_ret.reindex(strat.index)

# 持有期结束日期（用于把净值点画在期末）
end_dates = [rebal[rebal > d].min() for d in strat.index]

strat_curve = (1.0 + strat.values).cumprod()
spy_curve = (1.0 + spy_ret.fillna(0).values).cumprod()

x = [strat.index[0]] + list(end_dates)
strat_y = [1.0] + list(strat_curve)
spy_y = [1.0] + list(spy_curve)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, strat_y, marker="o", lw=2, label=f"Strategy (Top {TOP_N})")
ax.plot(x, spy_y, marker="s", lw=2, label="SPY (buy & hold)")
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_title("Strategy vs SPY - Growth of $1 (open-to-open, same periods)")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
out = "reports/strategy_vs_spy.png"
fig.savefig(out, dpi=120)

strat_total = strat_curve[-1] - 1
spy_total = spy_curve[-1] - 1
print(f"periods: {len(strat)}")
print(f"window:  {x[0].date()} -> {x[-1].date()}")
print(f"strategy total return: {strat_total:+.2%}")
print(f"SPY total return:      {spy_total:+.2%}")
print(f"saved: {out}")
