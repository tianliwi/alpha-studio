import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """除法，分母为 0 或缺失返回 NaN（不产生 inf）。"""
    b = b.replace(0, np.nan)
    return a / b


def compute_factors(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """从标准化基本面字段计算价值/质量因子。输入输出共享同一 MultiIndex（管线中为 (date, ticker)）。"""
    f = fundamentals
    out = pd.DataFrame(index=f.index)
    # 盈利能力
    out["roe"] = _safe_div(f["net_income"], f["total_equity"])
    out["roa"] = _safe_div(f["net_income"], f["total_assets"])
    out["gross_margin"] = _safe_div(f["gross_profit"], f["revenue"])
    out["net_margin"] = _safe_div(f["net_income"], f["revenue"])
    # 估值（用 yield 形式，越大越便宜，方向统一）
    out["earnings_yield"] = _safe_div(f["net_income"], f["market_cap"])
    out["fcf_yield"] = _safe_div(f["free_cash_flow"], f["market_cap"])
    out["pb"] = _safe_div(f["market_cap"], f["total_equity"])
    out["book_to_market"] = _safe_div(f["total_equity"], f["market_cap"])
    # 质量
    out["debt_to_equity"] = _safe_div(f["total_debt"], f["total_equity"])
    return out


# 进入模型的因子方向：True 表示值越大越看好
FACTOR_DIRECTION = {
    "roe": True, "roa": True, "gross_margin": True, "net_margin": True,
    "earnings_yield": True, "fcf_yield": True, "book_to_market": True,
    "pb": False, "debt_to_equity": False,
}
