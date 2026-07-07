import yfinance as yf

def fetch_market_data(
    ticker: str,
    start_date,
    end_date,
):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    return df