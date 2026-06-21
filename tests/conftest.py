import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_fundamentals():
    """两只股票、两个财报期的小型基本面数据。"""
    idx = pd.MultiIndex.from_tuples(
        [("2023-03-31", "AAA"), ("2023-03-31", "BBB"),
         ("2023-06-30", "AAA"), ("2023-06-30", "BBB")],
        names=["report_date", "ticker"],
    )
    return pd.DataFrame(
        {
            "net_income": [100.0, 50.0, 110.0, 40.0],
            "total_equity": [1000.0, 1000.0, 1000.0, 1000.0],
            "total_assets": [2000.0, 2500.0, 2000.0, 2500.0],
            "revenue": [500.0, 300.0, 520.0, 280.0],
            "gross_profit": [200.0, 90.0, 210.0, 84.0],
            "total_debt": [500.0, 1500.0, 500.0, 1500.0],
            "free_cash_flow": [80.0, 20.0, 85.0, 15.0],
            "market_cap": [5000.0, 2000.0, 5000.0, 2000.0],
            "shares_out": [100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )


@pytest.fixture
def sample_exec_prices():
    """两只股票 4 个调仓日的成交开盘价（exec_open），用于回测与未来收益。"""
    dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30"])
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    # AAA 稳定上涨，BBB 下跌
    exec_open = [10, 20, 11, 19, 12, 18, 13, 17]
    return pd.DataFrame({"exec_open": exec_open}, index=idx, dtype=float)
