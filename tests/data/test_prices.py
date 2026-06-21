from unittest.mock import patch
import pandas as pd
from alpha_studio.data import prices


def _fake_yf_download(tickers, **kwargs):
    dates = pd.to_datetime(["2023-01-31", "2023-02-01"])
    cols = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    # 列顺序: (Open,AAA)(Open,BBB)(Close,AAA)(Close,BBB)
    data = [[9.5, 99.0, 10.0, 100.0], [10.2, 109.0, 11.0, 110.0]]
    return pd.DataFrame(data, index=dates, columns=cols)


def test_fetch_prices_returns_open_and_close_long():
    with patch("alpha_studio.data.prices.yf.download", side_effect=_fake_yf_download):
        df = prices.fetch_prices(["AAA", "BBB"], "2023-01-01", "2023-03-01", use_cache=False)
    assert list(df.index.names) == ["date", "ticker"]
    assert "open" in df.columns and "close" in df.columns
    assert len(df) == 4  # 2 日 * 2 股
    assert df.loc[(pd.Timestamp("2023-01-31"), "AAA"), "close"] == 10.0
    assert df.loc[(pd.Timestamp("2023-01-31"), "AAA"), "open"] == 9.5


def test_execution_open_prices_uses_next_trading_day_open():
    # 调仓日 T 收盘算信号 → T+1 开盘成交
    dates = pd.to_datetime(["2023-01-31", "2023-02-01", "2023-02-28"])
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    daily = pd.DataFrame(
        {"open": [9.5, 10.2, 12.0], "close": [10.0, 11.0, 12.5]}, index=idx
    )
    rebal = pd.to_datetime(["2023-01-31"])
    exec_px = prices.execution_open_prices(daily, rebal)
    # T=2023-01-31 的成交价 = 下一交易日 2023-02-01 的开盘 10.2
    assert exec_px.loc[(pd.Timestamp("2023-01-31"), "AAA"), "exec_open"] == 10.2
