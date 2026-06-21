import numpy as np
import pandas as pd


def select_topn(scores: pd.DataFrame, top_n: int) -> pd.Series:
    """每个调仓日取打分最高的 top_n 只，等权。返回 MultiIndex(date, ticker) 权重 Series。"""
    parts = []
    for _, group in scores.groupby(level="date", sort=False):
        top = group["score"].nlargest(top_n)
        parts.append(pd.Series(1.0 / len(top), index=top.index))
    if not parts:
        return pd.Series(dtype=float, name="weight")
    return pd.concat(parts).rename("weight")


def run_backtest(weights: pd.Series, exec_prices: pd.DataFrame, cost: float) -> dict:
    """按权重逐期持有，open-to-open 计算扣成本后的组合收益序列与累计净值。

    weights: MultiIndex(date, ticker) 各调仓日目标权重（date=信号日 T）
    exec_prices: MultiIndex(date, ticker) 含 'exec_open'（T+1 成交开盘价）
    cost: 单边换手成本率
    """
    op = exec_prices["exec_open"].unstack("ticker").sort_index()
    period_ret = op.pct_change().shift(-1)  # 本次成交开盘 → 下次成交开盘

    w = weights.unstack("ticker").reindex(columns=op.columns).fillna(0.0).sort_index()
    rebal_dates = w.index

    gross_returns = []
    turnover_list = []
    prev_w = pd.Series(0.0, index=op.columns)
    for d in rebal_dates:
        cur_w = w.loc[d]
        turnover = (cur_w - prev_w).abs().sum()
        period_row = period_ret.loc[d]
        ret = np.nan if period_row.isna().all() else (cur_w * period_row).sum()
        net = ret - turnover * cost
        gross_returns.append(net)
        turnover_list.append(turnover)
        prev_w = cur_w

    raw_returns = pd.Series(gross_returns, index=rebal_dates, name="returns")
    valid = raw_returns.notna()
    returns = raw_returns[valid]
    equity = (1.0 + returns).cumprod()
    turnover = pd.Series(turnover_list, index=rebal_dates, name="turnover")[valid]
    return {"returns": returns, "equity_curve": equity, "turnover": turnover}
