import warnings
import pandas as pd
import yfinance as yf

from app.core.exceptions import ExternalDataFetchError, NoNewMarketDataError
from app.core.logging import get_logger

logger = get_logger(__name__)

def fetch_market_data(
    ticker: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )

            warning_messages = [
                str(w.message)
                for w in caught_warnings
            ]

    except Exception as e:
        message = str(e)

        logger.exception(
            "yfinance request failed | ticker=%s | start_date=%s | end_date=%s",
            ticker,
            start_date,
            end_date,
        )

        # rate limit / Yahoo side / network issue
        raise ExternalDataFetchError(
            f"Failed to fetch market data from yfinance for {ticker}: {message}"
        ) from e

    if df is None:
        logger.info(
            "No new market data returned by yfinance | ticker=%s | start_date=%s | end_date=%s | warnings=%s",
            ticker,
            start_date,
            end_date,
            warning_messages,
        )

        return {
                "status": "no_new_data",
                "ticker": ticker,
                "latest_before": end_date,
                "latest_after": end_date,
                "rows_fetched": 0,
                "rows_inserted_or_updated": 0,
                "message": f"No new market data available for {ticker}.",
            }
    
    return df

    
    
        