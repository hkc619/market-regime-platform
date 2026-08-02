from sqlalchemy import text
from sqlalchemy.orm import Session

def get_market_data(
    ticker: str, 
    db: Session
    ):

    ticker = ticker.upper().strip()

    query = text(
        """
        SELECT
            ticker,
            COUNT(*) AS row_count,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM market_prices
        WHERE ticker = :ticker
        GROUP BY ticker;
        """
    )

    row = db.execute(query, {"ticker": ticker}).mappings().first()

    return row


def get_macro_daily(db: Session):
    query = text(
        """
        SELECT
            COUNT(*) AS total_rows,
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            COUNT(vix) AS vix_count,
            COUNT(yield_10yr) AS yield_10yr_count,
            COUNT(yield_2yr) AS yield_2yr_count
        FROM macro_daily;
        """
    )

    row = db.execute(query).mappings().first()

    return row


def get_window(db: Session, lookback: int, ticker: str):
    
    ticker = ticker.upper().strip()
    
    query = text(
        """
        SELECT 
            ticker,
            COUNT(*) AS row_count,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM (
            SELECT ticker, date 
            FROM market_prices
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :lookback
        ) AS recent_prices
        GROUP BY ticker;
        
        """
        )
    
    row = db.execute(query, {
        "ticker": ticker, 
        "lookback": lookback
        }).mappings().first()
    
    return row
