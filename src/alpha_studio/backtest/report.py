import numpy as np
import pandas as pd
from loguru import logger

from alpha_studio.config import REPORTS_DIR

PERIODS_PER_YEAR = 12  # 月频


def performance_metrics(returns: pd.Series) -> dict:
    """从月频收益序列计算关键绩效指标。"""
    returns = returns.dropna()
    if returns.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan,
                "max_drawdown": np.nan, "win_rate": np.nan}
    ann_return = (1 + returns).prod() ** (PERIODS_PER_YEAR / len(returns)) - 1
    ann_vol = returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "annual_return": float(ann_return),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
    }


def save_tearsheet(returns: pd.Series, benchmark: pd.Series | None = None,
                   name: str = "backtest") -> None:
    """用 pyfolio 生成 tearsheet 图表并保存。绘图失败不影响主流程。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyfolio as pf

        pf.create_returns_tear_sheet(returns, benchmark_rets=benchmark, return_fig=True)
        out = REPORTS_DIR / f"{name}_tearsheet.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close("all")
        logger.info(f"saved tearsheet: {out}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"tearsheet skipped: {e}")
