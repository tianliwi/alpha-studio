from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data_cache"
PRICES_DIR = CACHE_DIR / "prices"
FUNDAMENTALS_DIR = CACHE_DIR / "fundamentals"
REPORTS_DIR = ROOT / "reports"

# 回测/策略默认参数
REBALANCE_FREQ = "ME"          # pandas month-end
TOP_N = 25                      # 每期持仓数
TRANSACTION_COST = 0.001        # 单边成本（手续费+滑点）
FUNDAMENTAL_LAG_DAYS = 60       # 财报发布滞后天数
BENCHMARK = "^GSPC"             # S&P 500 指数

for _d in (CACHE_DIR, PRICES_DIR, FUNDAMENTALS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
