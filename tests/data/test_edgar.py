import pandas as pd
import pytest

from alpha_studio.data import edgar


def _flow(start, end, val, **kw):
    return {"start": start, "end": end, "val": val, "filed": kw.get("filed", end), **kw}


def _instant(end, val, **kw):
    return {"end": end, "val": val, "filed": kw.get("filed", end)}


def test_cik_for_tickers_normalizes_symbols():
    js = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
        "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "Berkshire"},
    }
    out = edgar.cik_for_tickers(js)
    assert out["AAPL"] == 320193
    assert out["BRK-B"] == 1067983  # 点转横杠


def test_quarterly_flow_keeps_three_month_spans_and_drops_ytd():
    entries = [
        _flow("2023-01-01", "2023-03-31", 100),   # Q1, ~90d -> keep
        _flow("2023-01-01", "2023-06-30", 210),   # 6mo YTD -> drop
        _flow("2023-04-01", "2023-06-30", 110),   # Q2, ~90d -> keep
    ]
    q = edgar.quarterly_flow(entries)
    assert q[pd.Timestamp("2023-03-31")] == 100
    assert q[pd.Timestamp("2023-06-30")] == 110
    assert pd.Timestamp("2023-06-30") in q and len(q) == 2


def test_quarterly_flow_derives_q4_from_annual_minus_three_quarters():
    entries = [
        _flow("2023-01-01", "2023-03-31", 100),   # Q1
        _flow("2023-04-01", "2023-06-30", 110),   # Q2
        _flow("2023-07-01", "2023-09-30", 120),   # Q3
        _flow("2023-01-01", "2023-12-31", 500),   # FY annual -> Q4 = 500-330 = 170
    ]
    q = edgar.quarterly_flow(entries)
    assert q[pd.Timestamp("2023-12-31")] == 170


def test_quarterly_flow_dedup_prefers_latest_filed():
    entries = [
        _flow("2023-01-01", "2023-03-31", 100, filed="2023-04-15"),
        _flow("2023-01-01", "2023-03-31", 105, filed="2023-10-15"),  # 重述，更晚
    ]
    q = edgar.quarterly_flow(entries)
    assert q[pd.Timestamp("2023-03-31")] == 105


def test_instant_series_takes_balance_points():
    entries = [_instant("2023-03-31", 1000), _instant("2023-06-30", 1100)]
    s = edgar.instant_series(entries)
    assert s[pd.Timestamp("2023-03-31")] == 1000
    assert s[pd.Timestamp("2023-06-30")] == 1100


def test_build_frame_full_contract():
    facts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": [_flow("2023-01-01", "2023-03-31", 50)]}},
                "Revenues": {"units": {"USD": [_flow("2023-01-01", "2023-03-31", 500)]}},
                "GrossProfit": {"units": {"USD": [_flow("2023-01-01", "2023-03-31", 200)]}},
                "StockholdersEquity": {"units": {"USD": [_instant("2023-03-31", 1000)]}},
                "Assets": {"units": {"USD": [_instant("2023-03-31", 3000)]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_instant("2023-03-31", 400)]}},
                "DebtCurrent": {"units": {"USD": [_instant("2023-03-31", 100)]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_flow("2023-01-01", "2023-03-31", 90)]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_flow("2023-01-01", "2023-03-31", 30)]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [_instant("2023-03-31", 1_000_000)]}},
            },
        }
    }
    frame = edgar.build_fundamentals_frame("AAA", facts)
    row = frame.loc[(pd.Timestamp("2023-03-31"), "AAA")]
    assert row["net_income"] == 50
    assert row["revenue"] == 500
    assert row["gross_profit"] == 200
    assert row["total_equity"] == 1000
    assert row["total_assets"] == 3000
    assert row["total_debt"] == 500          # 400 + 100
    assert row["free_cash_flow"] == 60       # 90 - 30
    assert row["shares_out"] == 1_000_000
    assert list(frame.index.names) == ["report_date", "ticker"]


def test_build_frame_revenue_concept_fallback():
    # 老标签 SalesRevenueNet 也应被识别为 revenue
    facts = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [_flow("2023-01-01", "2023-03-31", 777)]}},
    }}}
    frame = edgar.build_fundamentals_frame("AAA", facts)
    assert frame.loc[(pd.Timestamp("2023-03-31"), "AAA")]["revenue"] == 777


def test_build_frame_empty_when_no_facts():
    assert edgar.build_fundamentals_frame("AAA", {"facts": {}}).empty


def test_build_frame_drops_shareonly_rows_and_ffills_shares():
    # 封面股数日期（2023-05-01）晚于期末、且无财务字段：应被丢弃，
    # 但其股数应前向填充影响不到更早的财报行；财报行自身保留股数。
    facts = {"facts": {
        "us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [_flow("2023-01-01", "2023-03-31", 50)]}},
            "Assets": {"units": {"USD": [_instant("2023-03-31", 3000)]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            _instant("2023-03-31", 1_000_000),
            _instant("2023-05-01", 1_010_000),   # 封面孤立行
        ]}}},
    }}
    frame = edgar.build_fundamentals_frame("AAA", facts)
    # 只剩一行财报行
    assert len(frame) == 1
    assert pd.Timestamp("2023-05-01") not in frame.index.get_level_values("report_date")
    assert frame.loc[(pd.Timestamp("2023-03-31"), "AAA")]["shares_out"] == 1_000_000
