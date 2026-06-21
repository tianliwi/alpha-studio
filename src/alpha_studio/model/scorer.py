import lightgbm as lgb
import numpy as np
import pandas as pd


def compute_forward_returns(exec_prices: pd.DataFrame) -> pd.Series:
    """每个调仓日的未来 1 期收益，open-to-open：下次成交开盘 / 本次成交开盘 - 1。

    exec_prices: MultiIndex(date, ticker)，列 'exec_open'（T+1 成交开盘价）。
    """
    op = exec_prices["exec_open"].unstack("ticker").sort_index()
    fwd = op.shift(-1) / op - 1.0
    return fwd.stack(future_stack=True).rename("fwd_return")


def _train_predict(train_X, train_y, pred_X) -> np.ndarray:
    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=3, num_leaves=7,
        learning_rate=0.05, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
        random_state=42,
    )
    model.fit(train_X, train_y)
    return model.predict(pred_X)


def walk_forward_score(factors: pd.DataFrame, fwd_returns: pd.Series,
                       min_train_dates: int = 12) -> pd.DataFrame:
    """walk-forward：对每个调仓日，用之前所有 (因子, 未来收益) 训练，预测当期打分。

    返回 MultiIndex(date, ticker)，单列 'score'。
    """
    data = factors.join(fwd_returns, how="inner")
    feature_cols = list(factors.columns)
    all_dates = sorted(data.index.get_level_values("date").unique())

    out_frames = []
    for i, d in enumerate(all_dates):
        if i < min_train_dates:
            continue
        train = data[data.index.get_level_values("date") < d].dropna(subset=["fwd_return"])
        train = train.dropna(subset=feature_cols)
        pred = factors.xs(d, level="date", drop_level=False).dropna(subset=feature_cols)
        if train.empty or pred.empty:
            continue
        preds = _train_predict(train[feature_cols], train["fwd_return"], pred[feature_cols])
        out_frames.append(pd.DataFrame({"score": preds}, index=pred.index))

    if not out_frames:
        return pd.DataFrame(columns=["score"])
    return pd.concat(out_frames).sort_index()
