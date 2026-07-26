from sqlalchemy.orm import Session
from datetime import date

from app.db.session import get_db
from app.core.logging import get_logger
from app.repositories.market_price_repository import (
    get_latest_ticker_prices, 
    get_latest_support_prices,
    get_ticker_window_ending_at
    )
from app.repositories.macro_repository import get_macro_daily, get_macro_monthly

def get_latest_ticker_window(
        ticker,
        db: Session,
        lookback: int = 312, 
):
    ticker = ticker.upper().strip()

    rows = get_latest_ticker_prices(
        db=db,
        ticker=ticker,
        lookback=lookback,
    )

    return list(reversed(rows))
    

def get_latest_support_window(
        support, 
        db: Session, 
        start_date,
        end_date):

    support = support.upper().strip()

    rows = get_latest_support_prices(
        db=db,
        support=support,
        start_date=start_date,
        end_date=end_date
    )

    return list(rows)

def get_macro_daily_window(db: Session):

    rows = get_macro_daily(db=db)

    return rows


def get_macro_monthly_window(db: Session):
    
    rows = get_macro_monthly(db=db)          

    return rows

def get_ticker_window_for_date(
        ticker,
        db: Session,
        as_of_date: date,
        lookback: int = 312, 
):
    ticker = ticker.upper().strip()

    rows = get_ticker_window_ending_at(
        db=db,
        ticker=ticker,
        as_of_date=as_of_date,
        lookback=lookback,
    )

    return list(reversed(rows))


# if __name__ == "__main__":
#     db = next(get_db())
#     try:
#         rows = get_ticker_window_for_date(
#             db=db,
#             ticker="SPY",
#             as_of_date=date(2025, 12, 1)
#         )
#         print(rows[0])
#         print(len(rows))
#         print(rows[-1])

#     finally:
#         db.close()


