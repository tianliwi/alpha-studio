"""SEC EDGAR companyfacts 基本面数据源。

免费、无需 API key、无 2 年历史上限。单次 companyfacts 请求即可拿到某公司
全部 XBRL 历史。本模块负责把杂乱的 us-gaap 标签归一化为与 yfinance 版本
相同的标准字段，输出契约一致：MultiIndex(report_date, ticker)，列为
net_income / revenue / gross_profit / total_equity / total_assets /
total_debt / free_cash_flow / shares_out。
"""
from __future__ import annotations

import hashlib
import time

import pandas as pd
import requests
from loguru import logger

from alpha_studio.config import FUNDAMENTALS_DIR

# SEC 要求声明带联系方式的 User-Agent
HEADERS = {"User-Agent": "alpha-studio research (contact: research@example.com)"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# 标准字段 -> us-gaap 概念别名（按优先级；用于跨年份标签迁移的容错）
FLOW_CONCEPTS = {
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
}
INSTANT_CONCEPTS = {
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_assets": ["Assets"],
}
# 现金流量（流量）——用于派生 free_cash_flow
OCF_CONCEPTS = ["NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX_CONCEPTS = ["PaymentsToAcquirePropertyPlantAndEquipment",
                  "PaymentsToAcquireProductiveAssets"]
# 总负债（时点）——长期 + 流动部分之和的容错组合
LT_DEBT_CONCEPTS = ["LongTermDebtNoncurrent", "LongTermDebt"]
ST_DEBT_CONCEPTS = ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"]
SHARES_CONCEPTS_DEI = ["EntityCommonStockSharesOutstanding"]
SHARES_CONCEPTS_GAAP = ["CommonStockSharesOutstanding"]

_QUARTER_MIN, _QUARTER_MAX = 80, 100
_ANNUAL_MIN, _ANNUAL_MAX = 350, 380


def cik_for_tickers(ticker_json: dict) -> dict:
    """把 SEC company_tickers.json 解析为 {ticker(大写,'-'风格): cik(int)}。"""
    out = {}
    for row in ticker_json.values():
        sym = str(row["ticker"]).upper().replace(".", "-").strip()
        out[sym] = int(row["cik_str"])
    return out


def _entries(facts: dict, namespace: str, concept: str, unit: str) -> list:
    try:
        return facts["facts"][namespace][concept]["units"][unit]
    except (KeyError, TypeError):
        return []


def _dedup_latest(entries: list, key) -> dict:
    """同一周期可能被多次申报（原始+重述），保留 filed 最新的一条。"""
    out: dict = {}
    for e in entries:
        k = key(e)
        if k is None:
            continue
        prev = out.get(k)
        if prev is None or str(e.get("filed", "")) >= str(prev.get("filed", "")):
            out[k] = e
    return out


def _span_days(e: dict) -> int | None:
    if "start" not in e or "end" not in e:
        return None
    return (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days


def quarterly_flow(entries: list) -> dict:
    """从某流量概念的 USD 条目里抽出"单季(3 个月)"值，按 end 日期返回 {Timestamp: val}。

    - 直接采用 span≈90 天的离散季度值（10-Q 通常含"三个月"列）。
    - Q4 在 10-K 只有全年值：用 全年 - 同一财年内三个季度之和 派生。
    """
    by_end_q = _dedup_latest(
        [e for e in entries if (_span_days(e) or 0) and _QUARTER_MIN <= _span_days(e) <= _QUARTER_MAX],
        key=lambda e: pd.Timestamp(e["end"]),
    )
    quarterly = {end: e["val"] for end, e in by_end_q.items()}

    annual = _dedup_latest(
        [e for e in entries if (_span_days(e) or 0) and _ANNUAL_MIN <= _span_days(e) <= _ANNUAL_MAX],
        key=lambda e: pd.Timestamp(e["end"]),
    )
    for end, e in annual.items():
        year_start = end - pd.Timedelta(days=_ANNUAL_MAX)
        in_year = [v for q, v in quarterly.items() if year_start < q <= end]
        if len(in_year) == 3:
            quarterly.setdefault(end, e["val"] - sum(in_year))
    return quarterly


def instant_series(entries: list) -> dict:
    """时点（资产负债表）概念：按 end 日期返回 {Timestamp: val}（取最新申报）。"""
    by_end = _dedup_latest(
        [e for e in entries if "end" in e and "start" not in e],
        key=lambda e: pd.Timestamp(e["end"]),
    )
    return {end: e["val"] for end, e in by_end.items()}


def _merge_priority(maps: list) -> dict:
    """按优先级合并多个 {end: val}：高优先级先填，低优先级补缺。"""
    out: dict = {}
    for m in maps:
        for k, v in m.items():
            out.setdefault(k, v)
    return out


def _flow_field(facts: dict, concepts: list) -> dict:
    return _merge_priority([quarterly_flow(_entries(facts, "us-gaap", c, "USD")) for c in concepts])


def _instant_field(facts: dict, concepts: list) -> dict:
    return _merge_priority([instant_series(_entries(facts, "us-gaap", c, "USD")) for c in concepts])


def _sum_fields(a: dict, b: dict) -> dict:
    """对齐 end 日期求和；任一缺失则该日期取另一个（视缺失为 0 不可取，故仅在两者皆有时相加，
    否则回退到存在的一方）。"""
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def build_fundamentals_frame(ticker: str, facts: dict) -> pd.DataFrame:
    """把单家公司的 companyfacts 解析为标准化季度长表（report_date=周期末）。"""
    fields: dict[str, dict] = {}
    for std, concepts in FLOW_CONCEPTS.items():
        fields[std] = _flow_field(facts, concepts)
    for std, concepts in INSTANT_CONCEPTS.items():
        fields[std] = _instant_field(facts, concepts)

    ocf = _flow_field(facts, OCF_CONCEPTS)
    capex = _flow_field(facts, CAPEX_CONCEPTS)
    fcf = {k: ocf[k] - capex[k] for k in ocf.keys() & capex.keys()}
    fields["free_cash_flow"] = fcf

    lt = _instant_field(facts, LT_DEBT_CONCEPTS)
    st = _instant_field(facts, ST_DEBT_CONCEPTS)
    fields["total_debt"] = _sum_fields(lt, st)

    shares = _merge_priority(
        [instant_series(_entries(facts, "dei", c, "shares")) for c in SHARES_CONCEPTS_DEI]
        + [instant_series(_entries(facts, "us-gaap", c, "shares")) for c in SHARES_CONCEPTS_GAAP]
    )
    fields["shares_out"] = shares

    financial_cols = ["net_income", "revenue", "gross_profit", "total_equity",
                      "total_assets", "total_debt", "free_cash_flow"]
    cols = financial_cols + ["shares_out"]
    frame = pd.DataFrame({c: pd.Series(fields.get(c, {})) for c in cols})
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    # EDGAR 的封面"流通股数"日期与报告期末不同，会产生只有 shares_out 的孤立行。
    # 先把股数前向填充到各财报行，再丢弃纯股数行，避免污染 as-of 对齐。
    frame["shares_out"] = pd.to_numeric(frame["shares_out"], errors="coerce").ffill()
    frame = frame.dropna(how="all", subset=financial_cols)
    frame.index.name = "report_date"
    frame["ticker"] = ticker
    frame = frame.set_index("ticker", append=True).sort_index()
    return frame


def fetch_company_facts(cik: int, session: requests.Session) -> dict:
    resp = session.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_cik_map(session: requests.Session) -> dict:
    resp = session.get(TICKERS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return cik_for_tickers(resp.json())


def fetch_fundamentals_edgar(tickers: list[str], use_cache: bool = True,
                             rate_limit_s: float = 0.15) -> pd.DataFrame:
    """从 EDGAR 拉取所有 ticker 的全历史季度基本面，输出契约同 yfinance 版本。"""
    key = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    cache = FUNDAMENTALS_DIR / f"edgar_fundamentals_{key}.parquet"
    if use_cache and cache.exists():
        logger.info(f"edgar fundamentals cache hit: {cache}")
        return pd.read_parquet(cache)

    session = requests.Session()
    cik_map = get_cik_map(session)
    frames = []
    missing = 0
    for i, t in enumerate(tickers, 1):
        cik = cik_map.get(t.upper())
        if cik is None:
            missing += 1
            logger.warning(f"no CIK for {t}")
            continue
        try:
            facts = fetch_company_facts(cik, session)
            frame = build_fundamentals_frame(t, facts)
            if not frame.empty:
                frames.append(frame)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"skip {t} (CIK {cik}): {e}")
        time.sleep(rate_limit_s)  # 守在 SEC 10 req/s 限速之下
        if i % 50 == 0:
            logger.info(f"edgar: {i}/{len(tickers)} fetched")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    logger.info(f"edgar fundamentals: {len(out)} rows, {len(frames)} tickers, {missing} missing CIK")
    if use_cache:
        out.to_parquet(cache)
    return out
