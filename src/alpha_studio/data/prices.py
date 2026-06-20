import pandas as pd
import yfinance as yf
from loguru import logger

from alpha_studio.config import PRICES_DIR


def _cache_path(start: str, end: str):
    return PRICES_DIR / f"prices_{start}_{end}.parquet"


def _field_long(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.Series:
    panel = raw[field]
    if isinstance(panel, pd.Series):  # 单只股票
        panel = panel.to_frame(tickers[0])
    long = panel.stack(future_stack=True).rename(field.lower())
    long.index = long.index.set_names(["date", "ticker"])
    return long


def fetch_prices(tickers: list[str], start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """拉取日线开盘价+收盘价，返回 MultiIndex(date, ticker) 长表，列含 'open'、'close'。"""
    cache = _cache_path(start, end)
    if use_cache and cache.exists():
        logger.info(f"price cache hit: {cache}")
        return pd.read_parquet(cache)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    open_ = _field_long(raw, "Open", tickers)
    close = _field_long(raw, "Close", tickers)
    long = pd.concat([open_, close], axis=1).dropna()

    if use_cache:
        long.to_parquet(cache)
    return long


def execution_open_prices(daily_prices: pd.DataFrame, rebalance_dates) -> pd.DataFrame:
    """对每个调仓日 T，取 T 之后第一个交易日的开盘价作为成交价（T 收盘算信号 → T+1 开盘成交）。

    返回 MultiIndex(date=调仓日 T, ticker)，列 'exec_open'。
    """
    opens = daily_prices["open"].unstack("ticker").sort_index()
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    rows = []
    for t in rebalance_dates:
        later = opens.index[opens.index > t]
        if len(later) == 0:
            continue
        exec_row = opens.loc[later[0]].rename("exec_open").to_frame()
        exec_row["date"] = t
        exec_row = exec_row.set_index("date", append=True).reorder_levels(["date", "ticker"])
        rows.append(exec_row)

    if not rows:
        return pd.DataFrame(columns=["exec_open"])
    out = pd.concat(rows).dropna().sort_index()
    out.index = out.index.set_names(["date", "ticker"])
    return out
