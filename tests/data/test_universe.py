from unittest.mock import patch
import pandas as pd
from alpha_studio.data import universe


def test_get_sp500_tickers_parses_wikipedia_table():
    fake_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B"]})
    with patch("alpha_studio.data.universe.pd.read_html", return_value=[fake_table]):
        tickers = universe.get_sp500_tickers()
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    # yfinance 用 '-' 而非 '.'
    assert "BRK-B" in tickers
    assert "BRK.B" not in tickers
