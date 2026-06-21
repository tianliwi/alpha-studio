# alpha-studio

A Buffett-style **value-investing alpha factor discovery + backtesting** system for US S&P 500 equities, with monthly rebalancing.

The pipeline: fetch universe → fetch prices (yfinance) + fundamentals (SEC EDGAR) → compute value/quality factors → cross-sectionally standardize → train a walk-forward LightGBM ranker → backtest open-to-open with transaction costs → report metrics + tearsheet.

**Execution model (no look-ahead):** a factor signal is computed at day **T's close**, the trade is executed at **T+1's open**. Forward-return labels and the backtest are both **open-to-open**. Walk-forward training uses a **1-period embargo** so the most recent label (which only fully realizes at the current execution price) is excluded.

---

## 1. Data structures

All tabular data uses pandas `MultiIndex` frames. Two index conventions appear:
- `report_date` — the fiscal-period-end date of a financial statement.
- `date` — a **rebalance/signal day T** (a trading day) used throughout the factor/score/backtest stages.

| Stage | Object | Index | Columns / dtype | Produced by |
|-------|--------|-------|-----------------|-------------|
| Universe | `list[str]` | — | yfinance-style tickers, e.g. `BRK-B` | `data/universe.py` |
| Daily prices | `DataFrame` | `(date, ticker)` | `open`, `close` (adjusted) | `data/prices.py` |
| Fundamentals (standardized) | `DataFrame` | `(report_date, ticker)` | `net_income`, `revenue`, `gross_profit`, `total_equity`, `total_assets`, `total_debt`, `free_cash_flow`, `shares_out` | `data/edgar.py` / `data/fundamentals.py` |
| Execution prices | `DataFrame` | `(date=T, ticker)` | `exec_open` (the T+1 open price) | `data/prices.py::execution_open_prices` |
| Aligned fundamentals | `DataFrame` | `(date, ticker)` | as-of-joined fundamentals visible at each T (report_date + lag ≤ T) | `factors/lag.py` |
| Factor matrix | `DataFrame` | `(date, ticker)` | `roe`, `roa`, `gross_margin`, `net_margin`, `earnings_yield`, `fcf_yield`, `pb`, `book_to_market`, `debt_to_equity` (cross-sectional z-scores, direction-normalized so "larger = better") | `factors/pipeline.py` |
| Forward returns | `Series` | `(date, ticker)` | `fwd_return` (open-to-open, next period) | `model/scorer.py` |
| Scores | `DataFrame` | `(date, ticker)` | `score` (LightGBM predicted forward return) | `model/scorer.py` |
| Weights | `Series` | `(date, ticker)` | `weight` (equal-weight Top-N each period) | `backtest/engine.py` |
| Backtest result | `dict` | — | `returns` (Series by T), `equity_curve` (Series), `turnover` (Series) | `backtest/engine.py` |
| Metrics | `dict` | — | `annual_return`, `annual_vol`, `sharpe`, `max_drawdown`, `win_rate` | `backtest/report.py` |

### Standardized fundamental fields

The 8 fields below are the canonical schema both data sources emit. EDGAR derives them from raw us-gaap XBRL tags:

| Field | Type | EDGAR derivation |
|-------|------|------------------|
| `net_income` | flow (quarterly) | `NetIncomeLoss` / `ProfitLoss` |
| `revenue` | flow | `RevenueFromContractWithCustomerExcludingAssessedTax` → `Revenues` → `SalesRevenueNet` (alias fallback) |
| `gross_profit` | flow | `GrossProfit` |
| `total_equity` | instant | `StockholdersEquity` |
| `total_assets` | instant | `Assets` |
| `total_debt` | instant | (`LongTermDebtNoncurrent`/`LongTermDebt`) + (`LongTermDebtCurrent`/`DebtCurrent`/`ShortTermBorrowings`) |
| `free_cash_flow` | flow | `NetCashProvidedByUsedInOperatingActivities` − `PaymentsToAcquirePropertyPlantAndEquipment` |
| `shares_out` | instant | `dei:EntityCommonStockSharesOutstanding` (forward-filled onto report rows) |

Flow fields are extracted as **single-quarter (3-month) values**; Q4 is derived as `FY − (Q1+Q2+Q3)`. `market_cap` is *not* stored — it's derived live in the pipeline as `close × shares_out` so valuation factors refresh with daily prices.

---

## 2. Code, file by file

```
src/alpha_studio/
├── config.py                  # All paths + strategy constants; auto-creates cache dirs
├── data/
│   ├── universe.py            # get_sp500_tickers(): scrape S&P 500 list from Wikipedia (sends User-Agent)
│   ├── prices.py              # fetch_prices() (yfinance open+close, parquet cache); execution_open_prices() (T+1 open)
│   ├── edgar.py               # SEC EDGAR companyfacts source: XBRL → standardized fundamentals (DEFAULT)
│   └── fundamentals.py        # fetch_fundamentals() dispatcher (edgar|yfinance); yfinance impl; normalize_statements()
├── factors/
│   ├── definitions.py         # compute_factors(): 9 value/quality ratios + FACTOR_DIRECTION
│   ├── lag.py                 # align_to_rebalance(): as-of join, report visible only after lag_days (no look-ahead)
│   └── pipeline.py            # build_from_raw(): fundamentals→lag→market_cap→factors→zscore; build_factor_matrix()
├── model/
│   └── scorer.py              # compute_forward_returns() (open-to-open); walk_forward_score() (LightGBM, 1-period embargo)
├── backtest/
│   ├── engine.py              # select_topn() (equal-weight Top-N); run_backtest() (open-to-open, transaction costs)
│   └── report.py              # performance_metrics(); save_tearsheet() (pyfolio PNG)
├── evaluation/
│   └── alphalens_eval.py      # evaluate_factor(): single-factor IC / IR diagnostics (Alphalens)
└── cli/
    └── main.py                # Typer app: fetch-data, eval-factors, rank, backtest, run-pipeline
```

**Key module notes**

- **`config.py`** — `REBALANCE_FREQ="ME"`, `TOP_N=25`, `TRANSACTION_COST=0.001`, `FUNDAMENTAL_LAG_DAYS=60`, `BENCHMARK="^GSPC"`, `FUNDAMENTALS_SOURCE="edgar"`. Cache directories are created on import.
- **`data/edgar.py`** — fetches each company's full XBRL history in **one** `companyfacts` request (free, no key, ~20 years). Normalizes tags with alias fallbacks, extracts quarterly flows + derives Q4, dedups restatements (keeps latest `filed`), drops cover-page share-only rows. Politely rate-limited under SEC's 10 req/s. Cache key prefix: `edgar_fundamentals_`.
- **`data/fundamentals.py`** — `fetch_fundamentals()` dispatches on `config.FUNDAMENTALS_SOURCE`. `"yfinance"` (`fetch_fundamentals_yf`) is the legacy ~2-year source kept for comparison.
- **`factors/lag.py`** — for each T, picks the most recent statement whose `report_date + FUNDAMENTAL_LAG_DAYS ≤ T`. This is the core no-look-ahead guarantee on the fundamental side.
- **`model/scorer.py`** — walk-forward: for rebalance i (after `min_train_dates=12` warm-up) train on all samples strictly before `all_dates[i-1]` (the embargo), predict scores for date i.
- **`cli/main.py`** — `_build_scores()` wires prices→factors→forward-returns→walk-forward scores; commands compose it.

**`scripts/plot_vs_spy.py`** — standalone utility: reproduces the strategy equity curve and overlays SPY buy-and-hold (identical open-to-open timing), saving `reports/strategy_vs_spy.png`.

---

## 3. Running it

### Install

```powershell
cd alpha-studio

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by execution policy, run once (current user):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# cmd.exe:        .\.venv\Scripts\activate.bat
# macOS / Linux:  python3 -m venv .venv && source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the package and the `sp` console command **inside the venv**. Once the venv is activated, `sp`, `python`, and `pytest` all resolve to it. Run `deactivate` to exit.

> Without activating, you can call the venv directly: `.\.venv\Scripts\sp backtest ...`
> or run as a module: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m alpha_studio.cli.main backtest ...`

### Commands

```powershell
# 1. Populate caches: S&P 500 prices (yfinance) + fundamentals (EDGAR).
#    Slow on first run; cached afterward. EDGAR ~5-15 min for the full universe.
sp fetch-data --start 2021-06-20 --end 2026-06-20

# 2. (Optional) Single-factor effectiveness diagnostics (Alphalens IC/IR).
sp eval-factors --start 2021-06-20 --end 2026-06-20

# 3. Backtest: prints performance metrics and saves a tearsheet.
sp backtest --start 2021-06-20 --end 2026-06-20 --topk 25

# 4. Current Top-N picks for a given rebalance month (default: latest).
sp rank --date 2026-05 --start 2021-06-20 --end 2026-06-20

# 5. Full pipeline: fetch -> backtest -> rank.
sp run-pipeline --start 2021-06-20 --end 2026-06-20

# 6. (Optional) Strategy vs SPY chart.
python scripts\plot_vs_spy.py
```

All commands accept `--start` / `--end` (`YYYY-MM-DD`). Caches make re-runs fast — only the first fetch hits the network.

> The commands above assume the venv is **activated** (so `sp` and `python` point into `.venv`). With the editable install, `import alpha_studio` works without setting `PYTHONPATH`.

### Switching the fundamentals source

Edit `src/alpha_studio/config.py`:

```python
FUNDAMENTALS_SOURCE = "edgar"     # full history (default, recommended)
# FUNDAMENTALS_SOURCE = "yfinance"  # ~2 years only
```

---

## 4. Generated files

Everything below is written under the project root and is **git-ignored** (`data_cache/`, `reports/`, `*.parquet`).

| Path | Created by | Contents |
|------|-----------|----------|
| `data_cache/prices/prices_{start}_{end}_{hash}.parquet` | `fetch_prices` | Daily open+close for the universe. `{hash}` = first 8 hex of MD5(sorted tickers). |
| `data_cache/fundamentals/edgar_fundamentals_{hash}.parquet` | `fetch_fundamentals_edgar` | Full-history standardized fundamentals (EDGAR). |
| `data_cache/fundamentals/fundamentals_{hash}.parquet` | `fetch_fundamentals_yf` | Legacy yfinance fundamentals (only if source switched). |
| `reports/backtest_tearsheet.png` | `backtest` / `save_tearsheet` | Pyfolio returns tearsheet (cumulative returns, drawdowns, monthly heatmap, etc.). |
| `reports/strategy_vs_spy.png` | `scripts/plot_vs_spy.py` | Growth-of-$1: strategy vs SPY over identical periods. |

**Cache semantics:** caches are keyed by the (sorted) ticker set — and, for prices, the date range. Same inputs → instant cache hit; different universe or price range → a new file. Delete a file to force a refetch. EDGAR fundamentals always store *full* history regardless of `--start`/`--end` (those only window the prices and the backtest).

### How the cache is wired (cache-aside)

You won't find the string `"data_cache"` anywhere outside `config.py` — the path is defined once as a constant and imported by name everywhere else. The flow:

1. **`config.py` defines the paths** (the only literal occurrence of `data_cache`):
   ```python
   ROOT = Path(__file__).resolve().parents[2]   # project root
   CACHE_DIR        = ROOT / "data_cache"
   PRICES_DIR       = CACHE_DIR / "prices"
   FUNDAMENTALS_DIR = CACHE_DIR / "fundamentals"
   ```
2. **Each data function imports the constant** (not the literal) and builds the parquet path with `pathlib`:
   ```python
   from alpha_studio.config import PRICES_DIR
   cache = PRICES_DIR / f"prices_{start}_{end}_{hash}.parquet"
   ```
3. **Read-or-fetch (the cache-aside pattern)** in `fetch_prices` / `fetch_fundamentals_edgar`:
   ```python
   if use_cache and cache.exists():
       return pd.read_parquet(cache)      # HIT  → read from disk, no network
   raw = yf.download(...)                  # MISS → fetch live
   long.to_parquet(cache)                 #        write to disk for next time
   return long
   ```

So the parquet files are referenced **indirectly through `PRICES_DIR` / `FUNDAMENTALS_DIR`** — `grep data_cache` only hits `config.py`, while `grep PRICES_DIR` shows the call sites. Call chain for `sp backtest`:

```
cli/main.py backtest() → _build_scores()
  ├─ fetch_prices()              → PRICES_DIR/prices_*.parquet        (read or fetch)
  └─ build_from_raw() → fetch_fundamentals()   # dispatches on FUNDAMENTALS_SOURCE
       └─ fetch_fundamentals_edgar() → FUNDAMENTALS_DIR/edgar_*.parquet (read or fetch)
```

Reading a cached parquet manually:

```python
import pandas as pd
df = pd.read_parquet("data_cache/prices/prices_2021-06-20_2026-06-20_b081e037.parquet")
```

---

## Known limitations

- **Survivorship bias:** uses the *current* S&P 500 membership, so delisted/bankrupt names are absent — this inflates backtested returns. A point-in-time constituent history would fix it.
- **Derived debt/FCF:** `total_debt` and `free_cash_flow` are composed from multiple XBRL concepts; when a company lacks the expected tags the field is `NaN` (tolerated by cross-sectional z-scoring).
- **Quarterly Q4:** derived as `FY − 9-month`; if intermediate quarters are missing for a fiscal year, that Q4 is skipped (the as-of join carries the prior quarter).

## Development

```powershell
python -m pytest -q          # 30 tests
```

Specs and the implementation plan live under `docs/superpowers/`.
