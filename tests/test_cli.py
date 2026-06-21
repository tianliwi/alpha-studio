from typer.testing import CliRunner
from alpha_studio.cli.main import app

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["run-pipeline", "rank", "fetch-data", "eval-factors", "backtest"]:
        assert cmd in result.output


def test_rebalance_dates_are_month_end():
    from alpha_studio.cli.main import _rebalance_dates
    dates = _rebalance_dates("2023-01-01", "2023-04-15")
    # 月末日期
    assert all(d.is_month_end for d in dates)
    assert len(dates) == 3  # 1月、2月、3月末


def test_snap_to_trading_days_picks_last_trading_day():
    import pandas as pd
    from alpha_studio.cli.main import _snap_to_trading_days
    daily_idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01"]), ["AAA"]],
        names=["date", "ticker"],
    )
    daily = pd.DataFrame({"open": [1, 2, 3], "close": [1, 2, 3]}, index=daily_idx, dtype=float)
    # 日历月末 2023-01-31 是交易日 → 自身；2023-02-28 无样本 → 回退到 02-01
    snapped = _snap_to_trading_days(pd.to_datetime(["2023-01-31", "2023-02-28"]), daily)
    assert pd.Timestamp("2023-01-31") in snapped
    assert pd.Timestamp("2023-02-01") in snapped
