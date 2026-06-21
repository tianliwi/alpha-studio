import numpy as np
import pandas as pd
from alpha_studio.backtest import engine


def test_select_topn_picks_highest_scores():
    idx = pd.MultiIndex.from_tuples(
        [("2023-01-31", "AAA"), ("2023-01-31", "BBB"), ("2023-01-31", "CCC")],
        names=["date", "ticker"],
    )
    scores = pd.DataFrame({"score": [0.3, 0.1, 0.2]}, index=idx)
    holdings = engine.select_topn(scores, top_n=2)
    picked = holdings.xs("2023-01-31", level="date").index.tolist()
    assert set(picked) == {"AAA", "CCC"}
    # 等权
    assert np.isclose(holdings.iloc[0], 0.5)


def test_backtest_returns_match_equal_weight(sample_exec_prices):
    # 持仓：每期都持有 AAA + BBB 等权
    dates = sample_exec_prices.index.get_level_values("date").unique()[:-1]
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    weights = pd.Series(0.5, index=idx, name="weight")

    result = engine.run_backtest(weights, sample_exec_prices, cost=0.0)
    # 第一期 open-to-open：AAA 10→11(+10%)，BBB 20→19(-5%)，等权 = +2.5%
    first_ret = result["returns"].iloc[0]
    assert np.isclose(first_ret, 0.025)


def test_backtest_applies_transaction_cost(sample_exec_prices):
    dates = sample_exec_prices.index.get_level_values("date").unique()[:-1]
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    weights = pd.Series(0.5, index=idx, name="weight")

    no_cost = engine.run_backtest(weights, sample_exec_prices, cost=0.0)["returns"].iloc[0]
    with_cost = engine.run_backtest(weights, sample_exec_prices, cost=0.01)["returns"].iloc[0]
    # 首期建仓换手 100%，成本拉低收益
    assert with_cost < no_cost


def test_backtest_drops_terminal_period_without_next_open(sample_exec_prices):
    dates = sample_exec_prices.index.get_level_values("date").unique()
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    weights = pd.Series(0.5, index=idx, name="weight")

    result = engine.run_backtest(weights, sample_exec_prices, cost=0.01)

    assert result["returns"].index.tolist() == dates[:-1].tolist()
    assert result["turnover"].index.tolist() == dates[:-1].tolist()


def test_performance_metrics():
    from alpha_studio.backtest import report
    rets = pd.Series([0.02, -0.01, 0.03, 0.01],
                     index=pd.date_range("2023-01-31", periods=4, freq="ME"))
    m = report.performance_metrics(rets)
    assert "annual_return" in m
    assert "sharpe" in m
    assert "max_drawdown" in m
    assert m["max_drawdown"] <= 0
