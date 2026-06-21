import io

import pandas as pd
import requests

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (alpha-studio research tool)"}


def get_sp500_tickers() -> list[str]:
    """从 Wikipedia 抓取当前 S&P 500 成分股，返回 yfinance 风格 ticker。"""
    resp = requests.get(SP500_WIKI_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    symbols = tables[0]["Symbol"].astype(str).tolist()
    return [s.replace(".", "-").strip() for s in symbols]
