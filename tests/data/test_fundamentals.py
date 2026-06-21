import pandas as pd
from alpha_studio.data import fundamentals


def test_normalize_statements_maps_fields():
    # 模拟 yfinance：行是科目，列是报告期
    periods = pd.to_datetime(["2023-03-31", "2023-06-30"])
    income = pd.DataFrame(
        {periods[0]: [100.0, 500.0, 200.0], periods[1]: [110.0, 520.0, 210.0]},
        index=["Net Income", "Total Revenue", "Gross Profit"],
    )
    balance = pd.DataFrame(
        {periods[0]: [1000.0, 2000.0, 500.0], periods[1]: [1000.0, 2000.0, 500.0]},
        index=["Stockholders Equity", "Total Assets", "Total Debt"],
    )
    cashflow = pd.DataFrame(
        {periods[0]: [80.0], periods[1]: [85.0]},
        index=["Free Cash Flow"],
    )
    out = fundamentals.normalize_statements("AAA", income, balance, cashflow)
    assert list(out.index.names) == ["report_date", "ticker"]
    row = out.loc[(pd.Timestamp("2023-03-31"), "AAA")]
    assert row["net_income"] == 100.0
    assert row["total_equity"] == 1000.0
    assert row["revenue"] == 500.0
    assert row["free_cash_flow"] == 80.0
