import hashlib

import pandas as pd
import yfinance as yf
from loguru import logger

from alpha_studio.config import FUNDAMENTALS_DIR

# yfinance 报表行名 → 标准化字段名
FIELD_MAP = {
    "Net Income": "net_income",
    "Total Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "Stockholders Equity": "total_equity",
    "Total Assets": "total_assets",
    "Total Debt": "total_debt",
    "Free Cash Flow": "free_cash_flow",
}


def _extract(statement: pd.DataFrame, wanted: dict) -> pd.DataFrame:
    rows = {}
    for raw_name, std_name in wanted.items():
        if raw_name in statement.index:
            rows[std_name] = statement.loc[raw_name]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)  # index = report periods


def normalize_statements(ticker, income, balance, cashflow) -> pd.DataFrame:
    """把三张 yfinance 报表合并为标准化字段的长表。"""
    parts = [_extract(s, FIELD_MAP) for s in (income, balance, cashflow)]
    merged = pd.concat(parts, axis=1)
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "report_date"
    merged["ticker"] = ticker
    merged = merged.set_index("ticker", append=True)
    return merged.sort_index()


def fetch_fundamentals(tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
    """逐只拉取季度财报，合并为标准化长表。单只失败则跳过。"""
    key = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    cache = FUNDAMENTALS_DIR / f"fundamentals_{key}.parquet"
    if use_cache and cache.exists():
        logger.info(f"fundamentals cache hit: {cache}")
        return pd.read_parquet(cache)

    frames = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            norm = normalize_statements(
                t, tk.quarterly_financials, tk.quarterly_balance_sheet, tk.quarterly_cashflow
            )
            # shares_out 取最新快照（market_cap 由 pipeline 按每日 close 派生）
            info = tk.fast_info
            norm["shares_out"] = getattr(info, "shares", None)
            frames.append(norm)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"skip {t}: {e}")
            continue

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    if use_cache:
        out.to_parquet(cache)
    return out
