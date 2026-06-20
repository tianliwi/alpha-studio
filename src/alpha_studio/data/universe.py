import pandas as pd

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    """从 Wikipedia 抓取当前 S&P 500 成分股，返回 yfinance 风格 ticker。"""
    tables = pd.read_html(SP500_WIKI_URL)
    symbols = tables[0]["Symbol"].astype(str).tolist()
    return [s.replace(".", "-").strip() for s in symbols]
