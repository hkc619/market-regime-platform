from sqlalchemy.orm import Session
from datetime import date

from app.repositories.market_price_repository import (
    get_row_number_between_start_end,
    get_ticker_window_range
)

def get_start_end_row_count(
    ticker,
    db: Session,
    start_date: date,
    end_date: date,
):
    ticker = ticker.upper().strip()

    rows_len = get_row_number_between_start_end(
        db=db,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    row_count = rows_len["row_count"]

    return row_count

def get_range_ticker_prices(
    db: Session,
    ticker: str,
    start_date: date,
    end_date: date,
    lookback: int = 312
):
    start_to_end = get_start_end_row_count(
        ticker=ticker,
        db=db,
        start_date=start_date,
        end_date=end_date,
    )
    print("start to end: ", start_to_end)
    range = start_to_end + lookback

    rows = get_ticker_window_range(
        db=db,
        ticker=ticker,
        end_date=end_date,
        range=range
    )

    return list(reversed(rows))
