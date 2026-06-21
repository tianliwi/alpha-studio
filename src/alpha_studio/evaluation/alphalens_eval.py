import pandas as pd
from loguru import logger

from alpha_studio.config import REPORTS_DIR


def evaluate_factor(factor: pd.Series, prices: pd.DataFrame, name: str,
                    quantiles: int = 5) -> dict:
    """用 Alphalens 评估单因子的 IC 与分组收益。返回 IC 摘要 dict。

    factor: MultiIndex(date, ticker) 因子值
    prices: MultiIndex(date, ticker) 含 'close'
    """
    try:
        import alphalens as al

        pricing = prices["close"].unstack("ticker").sort_index()
        factor_data = al.utils.get_clean_factor_and_forward_returns(
            factor, pricing, quantiles=quantiles, periods=(1,),
        )
        ic = al.performance.factor_information_coefficient(factor_data)
        summary = {"ic_mean": float(ic.mean().iloc[0]), "ic_std": float(ic.std().iloc[0])}
        summary["ir"] = summary["ic_mean"] / summary["ic_std"] if summary["ic_std"] else float("nan")
        logger.info(f"factor {name}: IC={summary['ic_mean']:.4f} IR={summary['ir']:.3f}")
        return summary
    except Exception as e:  # noqa: BLE001
        logger.warning(f"alphalens eval failed for {name}: {e}")
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ir": float("nan")}
