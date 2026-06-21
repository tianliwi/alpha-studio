import pandas as pd

from alpha_studio.data.prices import fetch_prices
from alpha_studio.data.fundamentals import fetch_fundamentals
from alpha_studio.factors.definitions import compute_factors, FACTOR_DIRECTION
from alpha_studio.factors.lag import align_to_rebalance
from alpha_studio.config import FUNDAMENTAL_LAG_DAYS


def cross_sectional_zscore(factors: pd.DataFrame) -> pd.DataFrame:
    """对每个调仓日(date)内做横截面 z-score 标准化。

    注意：当某调仓日某因子全部相等（或仅一只股票）时 std=0，得到 0/0=NaN（非 inf），
    这是预期行为——下游模型/合成会自然跳过 NaN。
    """
    def _z(group):
        return (group - group.mean()) / group.std(ddof=0)
    return factors.groupby(level="date", group_keys=False).apply(_z)


def apply_direction(zscores: pd.DataFrame, direction: dict) -> pd.DataFrame:
    """按因子方向翻转：方向为 False 的因子乘 -1，使所有因子'越大越好'。"""
    out = zscores.copy()
    for col, positive in direction.items():
        if col in out.columns and not positive:
            out[col] = -out[col]
    return out


def build_factor_matrix(factors: pd.DataFrame) -> pd.DataFrame:
    """标准化 + 方向归一，输出可直接喂模型的因子矩阵。"""
    z = cross_sectional_zscore(factors)
    return apply_direction(z, FACTOR_DIRECTION)


def attach_market_cap(aligned_fund: pd.DataFrame, daily_prices: pd.DataFrame) -> pd.DataFrame:
    """对齐后的财报帧合并当日 close，派生 market_cap = close * shares_out（随价格每日刷新）。

    aligned_fund / daily_prices 均为 MultiIndex(date, ticker)。
    """
    close = daily_prices["close"].rename("close")
    out = aligned_fund.join(close, how="left")
    out["market_cap"] = out["close"] * out["shares_out"]
    return out


def build_from_raw(tickers, start, end, rebalance_dates, use_cache=True) -> pd.DataFrame:
    """端到端：拉基本面→滞后对齐→按当日 close 派生 market_cap→算因子→标准化。

    估值因子用调仓日 T 的 close 派生 market_cap（每日刷新），符合"T 收盘算信号"。
    返回 MultiIndex(date, ticker) 因子矩阵。

    要求：`rebalance_dates` 必须是交易日（调用方应先用 _snap_to_trading_days 对齐），
    否则当日无 close 行会导致 market_cap 全为 NaN。
    """
    fund = fetch_fundamentals(tickers, use_cache=use_cache)
    aligned = align_to_rebalance(fund, rebalance_dates, FUNDAMENTAL_LAG_DAYS)
    daily = fetch_prices(tickers, start, end, use_cache=use_cache)
    with_cap = attach_market_cap(aligned, daily)
    raw_factors = compute_factors(with_cap)
    return build_factor_matrix(raw_factors)
