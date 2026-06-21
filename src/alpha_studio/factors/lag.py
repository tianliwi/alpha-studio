import pandas as pd


def align_to_rebalance(factors: pd.DataFrame, rebalance_dates, lag_days: int) -> pd.DataFrame:
    """把按 report_date 的因子对齐到调仓日，财报延后 lag_days 才可见（无未来函数）。

    对每个调仓日，取该日前已可见（report_date + lag_days <= 调仓日）的最近一期财报。
    返回 MultiIndex(date, ticker)。
    """
    f = factors.reset_index()
    f["report_date"] = pd.to_datetime(f["report_date"])
    f["available_date"] = f["report_date"] + pd.Timedelta(days=lag_days)
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    rows = []
    for ticker, grp in f.groupby("ticker"):
        grp = grp.sort_values("available_date")
        for d in rebalance_dates:
            visible = grp[grp["available_date"] <= d]
            if visible.empty:
                continue
            latest = visible.iloc[-1]
            rec = latest.drop(["report_date", "available_date", "ticker"]).to_dict()
            rec["date"] = d
            rec["ticker"] = ticker
            rows.append(rec)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return out
