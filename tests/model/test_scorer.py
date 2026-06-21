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


def test_walk_forward_embargoes_most_recent_period(monkeypatch):
    # 严格无未来函数：预测某调仓日时，训练集必须剔除标签结束于本期成交价的最近一期
    factors, fwd = _make_dataset(n_dates=8, n_stocks=5)
    all_dates = sorted(factors.index.get_level_values("date").unique())

    captured = []
    real_train_predict = scorer._train_predict

    def _spy(train_X, train_y, pred_X):
        captured.append(train_X.index.get_level_values("date").max())
        return real_train_predict(train_X, train_y, pred_X)

    monkeypatch.setattr(scorer, "_train_predict", _spy)
    scorer.walk_forward_score(factors, fwd, min_train_dates=4)

    # 首次打分在 all_dates[4]；embargo 剔除 all_dates[3]，故最大训练日 = all_dates[2]
    assert captured[0] == all_dates[2]
    # 训练集绝不能包含被预测日自身或其紧邻的前一期
    assert captured[0] < all_dates[3]
