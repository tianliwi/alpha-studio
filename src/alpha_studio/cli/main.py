import pandas as pd
import typer
from loguru import logger

from alpha_studio.config import REBALANCE_FREQ, TOP_N, TRANSACTION_COST
from alpha_studio.data.universe import get_sp500_tickers
from alpha_studio.data.prices import fetch_prices, execution_open_prices
from alpha_studio.data.fundamentals import fetch_fundamentals
from alpha_studio.factors.pipeline import build_from_raw
from alpha_studio.model.scorer import compute_forward_returns, walk_forward_score
from alpha_studio.backtest.engine import select_topn, run_backtest
from alpha_studio.backtest.report import performance_metrics, save_tearsheet

app = typer.Typer(help="股票价值投资 alpha 因子发掘系统")


def _rebalance_dates(start: str, end: str) -> pd.DatetimeIndex:
    """日历月末（信号日候选）。实际使用时由 _snap_to_trading_days 对齐到交易日。"""
    return pd.date_range(start, end, freq=REBALANCE_FREQ)


def _snap_to_trading_days(calendar_dates, daily_prices) -> pd.DatetimeIndex:
    """把日历月末对齐到 <= 该日的最后一个交易日（信号日 T）。"""
    trading = daily_prices.index.get_level_values("date").unique().sort_values()
    snapped = []
    for d in calendar_dates:
        prior = trading[trading <= d]
        if len(prior):
            snapped.append(prior[-1])
    return pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(snapped)))


def _build_scores(tickers, start, end):
    """返回 (scores, exec_prices)。信号日 T 收盘算因子，exec_prices 为 T+1 开盘成交价。"""
    daily = fetch_prices(tickers, start, end)
    rebal = _snap_to_trading_days(_rebalance_dates(start, end), daily)
    factors = build_from_raw(tickers, start, end, rebal)
    exec_prices = execution_open_prices(daily, rebal)
    fwd = compute_forward_returns(exec_prices)
    scores = walk_forward_score(factors, fwd, min_train_dates=12)
    return scores, exec_prices


@app.command()
def fetch_data(start: str = "2018-01-01", end: str = "2024-12-31"):
    """仅更新数据缓存。"""
    tickers = get_sp500_tickers()
    fetch_prices(tickers, start, end)
    fetch_fundamentals(tickers)
    logger.info("data cache updated")


@app.command()
def eval_factors(start: str = "2018-01-01", end: str = "2024-12-31"):
    """跑 Alphalens 因子有效性诊断（按调仓日收盘价）。"""
    from alpha_studio.evaluation.alphalens_eval import evaluate_factor
    tickers = get_sp500_tickers()
    daily = fetch_prices(tickers, start, end)
    rebal = _snap_to_trading_days(_rebalance_dates(start, end), daily)
    factors = build_from_raw(tickers, start, end, rebal)
    # alphalens 评估用调仓日收盘价（因子在 T 收盘可得）
    close_long = daily.loc[
        daily.index.get_level_values("date").isin(rebal), ["close"]
    ].sort_index()
    for col in factors.columns:
        evaluate_factor(factors[col].dropna(), close_long, col)


@app.command()
def rank(date: str = typer.Option(None, help="调仓月，如 2024-06；默认最近"),
         start: str = "2018-01-01", end: str = "2024-12-31"):
    """输出某月当期排名股票清单。"""
    tickers = get_sp500_tickers()
    scores, _ = _build_scores(tickers, start, end)
    if scores.empty:
        typer.echo("无打分结果")
        raise typer.Exit(1)
    score_dates = scores.index.get_level_values("date").unique()
    if not date:
        target = score_dates.max()
    else:
        month = pd.Period(date, freq="M")
        in_month = [d for d in score_dates if pd.Period(d, freq="M") == month]
        if not in_month:
            typer.echo(f"{date} 无打分数据")
            raise typer.Exit(1)
        target = max(in_month)
    day = scores.xs(target, level="date")["score"].nlargest(TOP_N)
    typer.echo(f"== {target.date()} Top {TOP_N} ==")
    for i, (tk, sc) in enumerate(day.items(), 1):
        typer.echo(f"{i:2d}. {tk:6s} {sc:+.4f}")


@app.command()
def backtest(start: str = "2018-01-01", end: str = "2024-12-31", topk: int = TOP_N):
    """仅回测并输出绩效指标 + tearsheet。"""
    tickers = get_sp500_tickers()
    scores, exec_prices = _build_scores(tickers, start, end)
    weights = select_topn(scores, topk)
    result = run_backtest(weights, exec_prices, TRANSACTION_COST)
    metrics = performance_metrics(result["returns"])
    typer.echo("== 回测绩效 ==")
    for k, v in metrics.items():
        typer.echo(f"{k:16s}: {v:+.4f}")
    save_tearsheet(result["returns"], name="backtest")


@app.command()
def run_pipeline(start: str = "2018-01-01", end: str = "2024-12-31", topk: int = TOP_N):
    """全流程：拉数据→算因子→训练→回测→出报告。"""
    fetch_data(start, end)
    backtest(start, end, topk)
    rank(None, start, end)


if __name__ == "__main__":
    app()
