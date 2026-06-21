import numpy as np
import pandas as pd
from alpha_studio.factors import pipeline


def test_cross_sectional_zscore_per_date():
    idx = pd.MultiIndex.from_tuples(
        [("2023-05-31", "AAA"), ("2023-05-31", "BBB"), ("2023-05-31", "CCC")],
        names=["date", "ticker"],
    )
    df = pd.DataFrame({"roe": [0.1, 0.2, 0.3], "pb": [5.0, 3.0, 1.0]}, index=idx)
    out = pipeline.cross_sectional_zscore(df)
    # 每个调仓日内均值≈0
    day = out.loc["2023-05-31"]
    assert np.isclose(day["roe"].mean(), 0.0, atol=1e-9)
    assert np.isclose(day["roe"].std(ddof=0), 1.0, atol=1e-9)


def test_apply_direction_flips_negative_factors():
    idx = pd.MultiIndex.from_tuples(
        [("2023-05-31", "AAA"), ("2023-05-31", "BBB")],
        names=["date", "ticker"],
    )
    # pb 方向为 False（越小越好），应被翻转
    z = pd.DataFrame({"roe": [1.0, -1.0], "pb": [1.0, -1.0]}, index=idx)
    out = pipeline.apply_direction(z, {"roe": True, "pb": False})
    assert out.loc[("2023-05-31", "AAA"), "pb"] == -1.0  # 翻转后高 pb 变低分
    assert out.loc[("2023-05-31", "AAA"), "roe"] == 1.0


def test_attach_market_cap_from_daily_close():
    # 对齐后的财报帧（含 shares_out），按调仓日 close 派生 market_cap
    fund_idx = pd.MultiIndex.from_tuples(
        [("2023-05-31", "AAA")], names=["date", "ticker"]
    )
    aligned = pd.DataFrame({"net_income": [100.0], "shares_out": [100.0]}, index=fund_idx)
    daily_idx = pd.MultiIndex.from_tuples(
        [("2023-05-31", "AAA")], names=["date", "ticker"]
    )
    daily = pd.DataFrame({"open": [49.0], "close": [50.0]}, index=daily_idx)
    out = pipeline.attach_market_cap(aligned, daily)
    # market_cap = close * shares_out = 50 * 100 = 5000
    assert out.loc[("2023-05-31", "AAA"), "market_cap"] == 5000.0
