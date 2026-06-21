import numpy as np
import pandas as pd
from alpha_studio.model import scorer


def _make_dataset(n_dates=8, n_stocks=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_dates, freq="ME")
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    roe = rng.normal(size=len(idx))
    pb = rng.normal(size=len(idx))
    factors = pd.DataFrame({"roe": roe, "pb": pb}, index=idx)
    # 未来收益与 roe 正相关，便于验证模型学到信号
    fwd = 0.05 * factors["roe"] + rng.normal(scale=0.01, size=len(idx))
    return factors, fwd.rename("fwd_return")


def test_compute_forward_returns_open_to_open():
    # 成交开盘价已知，未来收益 = 下次成交开盘 / 本次成交开盘 - 1
    dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    exec_prices = pd.DataFrame({"exec_open": [10.0, 11.0, 11.0]}, index=idx)
    fwd = scorer.compute_forward_returns(exec_prices)
    # 2023-01-31 的未来收益 = 11/10 - 1 = 0.1
    assert np.isclose(fwd.loc[(pd.Timestamp("2023-01-31"), "AAA")], 0.1)
    # 最后一期无未来收益
    assert (pd.Timestamp("2023-03-31"), "AAA") not in fwd.dropna().index


def test_walk_forward_scores_align_with_signal():
    factors, fwd = _make_dataset()
    scores = scorer.walk_forward_score(factors, fwd, min_train_dates=4)
    # 只在第 5 个调仓日起才有打分（前 4 期用于训练）
    scored_dates = scores.index.get_level_values("date").unique()
    assert len(scored_dates) == 4
    # 在最后一个调仓日，roe 高的股票打分应整体更高（rank 相关为正）
    last = scores.index.get_level_values("date").max()
    merged = pd.concat([scores.xs(last, level="date"),
                        factors.xs(last, level="date")["roe"]], axis=1).dropna()
    corr = merged["score"].corr(merged["roe"], method="spearman")
    assert corr > 0.3
