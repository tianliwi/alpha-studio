import pandas as pd
from alpha_studio.factors import lag


def test_lag_factors_no_lookahead():
    # AAA 在 2023-03-31 发布财报，滞后 60 天 → 2023-05-30 才可见
    idx = pd.MultiIndex.from_tuples(
        [("2023-03-31", "AAA"), ("2023-06-30", "AAA")],
        names=["report_date", "ticker"],
    )
    factors = pd.DataFrame({"roe": [0.1, 0.2]}, index=idx)
    rebal_dates = pd.to_datetime(["2023-04-30", "2023-05-31", "2023-06-30"])

    aligned = lag.align_to_rebalance(factors, rebal_dates, lag_days=60)

    # 2023-04-30：财报尚不可见（3-31 + 60 = 5-30）→ 无数据
    assert ("2023-04-30", "AAA") not in aligned.index
    # 2023-05-31：3-31 财报已可见 → roe = 0.1
    assert aligned.loc[(pd.Timestamp("2023-05-31"), "AAA"), "roe"] == 0.1
    # 2023-06-30：6-30 财报刚发布尚未可见，仍用 3-31 的 0.1
    assert aligned.loc[(pd.Timestamp("2023-06-30"), "AAA"), "roe"] == 0.1
