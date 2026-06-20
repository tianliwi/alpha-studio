import numpy as np
from alpha_studio.factors import definitions as fd


def test_compute_factors_known_values(sample_fundamentals):
    out = fd.compute_factors(sample_fundamentals)
    # AAA 2023-03-31: ROE = net_income/total_equity = 100/1000 = 0.1
    aaa = out.loc[("2023-03-31", "AAA")]
    assert np.isclose(aaa["roe"], 0.1)
    # ROA = 100/2000 = 0.05
    assert np.isclose(aaa["roa"], 0.05)
    # gross_margin = 200/500 = 0.4
    assert np.isclose(aaa["gross_margin"], 0.4)
    # debt_to_equity = 500/1000 = 0.5
    assert np.isclose(aaa["debt_to_equity"], 0.5)
    # fcf_yield = free_cash_flow/market_cap = 80/5000 = 0.016
    assert np.isclose(aaa["fcf_yield"], 0.016)
    # pb = market_cap/total_equity = 5000/1000 = 5.0
    assert np.isclose(aaa["pb"], 5.0)
    # earnings_yield = net_income/market_cap = 100/5000 = 0.02 (PE 倒数)
    assert np.isclose(aaa["earnings_yield"], 0.02)


def test_compute_factors_handles_zero_denominator():
    import pandas as pd
    idx = pd.MultiIndex.from_tuples([("2023-03-31", "ZZZ")], names=["report_date", "ticker"])
    df = pd.DataFrame(
        {"net_income": [10.0], "total_equity": [0.0], "total_assets": [100.0],
         "revenue": [0.0], "gross_profit": [0.0], "total_debt": [0.0],
         "free_cash_flow": [5.0], "market_cap": [100.0], "shares_out": [10.0]},
        index=idx,
    )
    out = fd.compute_factors(df)
    # 除零应得 NaN 而非 inf
    assert np.isnan(out.loc[("2023-03-31", "ZZZ"), "roe"])
