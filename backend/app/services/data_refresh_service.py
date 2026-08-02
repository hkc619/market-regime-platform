
from sqlalchemy.orm import Session
from datetime import date, timedelta
import pandas as pd

from app.repositories.data_refresh_repository import get_latest_ohlcv, upsert_market_prices
from app.repositories.data_update_log_repository import create_data_update_log

from app.providers.market_data_normalizer import normalize_market_data
from app.core.logging import get_logger
from app.core.exceptions import NoNewMarketDataError, InvalidExternalDataError

import app.providers.yfinance_provider as YFinanceProvider

logger = get_logger(__name__)

def get_latest_market_date(
        ticker: str, 
        db:Session,
        request_id: str | None = None,
    ) -> dict:

    ticker = ticker.upper().strip()

    logger.info(
        "Market data refresh started | request_id=%s | ticker=%s",
        request_id,
        ticker,
    )

    latest_before = get_latest_ohlcv(
        ticker=ticker, 
        db=db
    )

    if latest_before is None:
        raise ValueError(f"No existing market data found for ticker={ticker}")
    
    return latest_before


def refresh_market_data(
        ticker: str, 
        db:Session,
        request_id: str | None = None,
        provider=None,
    ) -> dict:

    ticker = ticker.upper().strip()

    latest_before = get_latest_ohlcv(
        ticker=ticker, 
        db=db
    )

    if latest_before is None:
        raise ValueError(f"No existing market data found for ticker={ticker}")

    start_date = latest_before + timedelta(days=1)
    end_date = date.today() + timedelta(days=1)

    if start_date > end_date:
        create_data_update_log(
            db=db,
            request_id=request_id,
            data_type="market_price",
            source="yfinance",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            status="up_to_date",
            rows_fetched=0,
            rows_inserted_or_updated=0,
            error_message=None,
        )
        
        return {
            "ticker": ticker,
            "latest_before": latest_before,
            "latest_after": latest_before,
            "rows_fetched": 0,
            "rows_inserted_or_updated": 0,
            "status": "up_to_date",
        }
    
    provider = provider or YFinanceProvider()
    
    try:
        raw_df = provider.fetch_market_data(
            ticker=ticker, 
            start_date=start_date, 
            end_date=end_date
        )

    except NoNewMarketDataError as e:
        logger.info(
            "No new market data available | request_id=%s | ticker=%s | start_date=%s | end_date=%s",
            request_id,
            ticker,
            start_date,
            end_date,
        )

        create_data_update_log(
            db=db,
            request_id=request_id,
            data_type="market_price",
            source="yfinance",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            status="no_new_data",
            rows_fetched=0,
            rows_inserted_or_updated=0,
            error_message=e.message,
        )

        return {
            "ticker": ticker,
            "latest_before": latest_before,
            "latest_after": latest_before,
            "rows_fetched": 0,
            "rows_inserted_or_updated": 0,
            "status": "no_new_data",
            "message": e.message,
        }

    rows = normalize_market_data(
        df=raw_df,
        ticker=ticker,
        source="yfinance",
    )

    if not rows:
        logger.info(
            "Normalized market data is empty | request_id=%s | ticker=%s",
            request_id,
            ticker,
        )

        return {
            "ticker": ticker,
            "latest_before": latest_before,
            "latest_after": latest_before,
            "rows_fetched": 0,
            "rows_inserted_or_updated": 0,
            "status": "no_new_data",
            "message": "Fetched data contained no usable market rows.",
        }

    affected_rows = upsert_market_prices(
        db=db,
        rows=rows,
    )

    latest_after = get_latest_market_date(
        db=db,
        ticker=ticker,
    )

    create_data_update_log(
        db=db,
        request_id=request_id,
        data_type="market_price",
        source="yfinance",
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        status="success",
        rows_fetched=len(rows),
        rows_inserted_or_updated=affected_rows,
        error_message=None,
    )

    logger.info(
        "Market data refresh completed | request_id=%s | ticker=%s | rows=%s | latest_before=%s | latest_after=%s",
        request_id,
        ticker,
        affected_rows,
        latest_before,
        latest_after,
    )

    return {
        "ticker": ticker,
        "latest_before": latest_before,
        "latest_after": latest_after,
        "rows_fetched": len(rows),
        "rows_inserted_or_updated": affected_rows,
        "status": "success",
        "message": "Market data refreshed successfully.",
    }
    
REQUIRED_MARKET_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

def validate_market_data_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    missing_columns = [
        col for col in REQUIRED_MARKET_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise InvalidExternalDataError(
            f"Market data is missing required columns: {missing_columns}"
        )

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise InvalidExternalDataError(
            "Market data contains invalid date values."
        )

    price_columns = ["open", "high", "low", "close"]

    for col in price_columns:
        if (df[col] <= 0).any():
            raise InvalidExternalDataError(
                f"Market data contains non-positive values in column: {col}"
            )

    if (df["volume"] < 0).any():
        raise InvalidExternalDataError(
            "Market data contains negative volume values."
        )

    return df
