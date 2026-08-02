from sqlalchemy.orm import Session
from datetime import date

from app.repositories.market_price_repository import (
    get_rows_between_start_end,
    get_ticker_window_range
)

def get_start_end_rows(
    ticker,
    db: Session,
    start_date: date,
    end_date: date,
):
    ticker = ticker.upper().strip()

    rows = get_rows_between_start_end(
        db=db,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    return list(reversed(rows))

def get_range_ticker_prices(
    db: Session,
    ticker: str,
    start_date: date,
    end_date: date,
    lookback: int = 312
):
    start_to_end = get_start_end_rows(
        ticker=ticker,
        db=db,
        start_date=start_date,
        end_date=end_date,
    )
    
    print("start to end: ", len(start_to_end))
    range = len(start_to_end) + lookback

    rows = get_ticker_window_range(
        db=db,
        ticker=ticker,
        end_date=end_date,
        range=range
    )

    return list(reversed(rows))
    