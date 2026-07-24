from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.repositories.market_price_repository import get_latest_ticker_prices, get_latest_support_prices
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


# if __name__ == "__main__":
#     db = next(get_db())
#     try:
#         rows = get_latest_support_window(
#             db=db,
#             support="TLT",
#             start_date='2024-12-30',
#             end_date='2025-12-31'
#         )

#         print(len(rows))
#         print(rows[:3])

#     finally:
#         db.close()


