from datetime import date
from sqlalchemy import text
from sqlalchemy.orm import Session


def get_latest_ticker_prices(
    db: Session,
    ticker: str,
    lookback: int,
): 
    query = text(
    """
    SELECT 
        ticker,
        date,
        open,
        high,
        low,
        close,
        adjusted_close,
        volume
    FROM market_prices
    WHERE ticker = :ticker
    ORDER BY date DESC
    LIMIT :lookback;
    """)

    rows = db.execute(
        query, 
        {
        "ticker": ticker, 
        "lookback": lookback
        }
    ).mappings().all()

    return rows

def get_latest_support_prices(
    db: Session,
    support: str,
    start_date,
    end_date
): 
    query = text(
    """
    SELECT 
        ticker, 
        date, 
        close
    FROM market_prices
    WHERE ticker = :ticker 
    AND date BETWEEN :start_date AND :end_date
    ORDER BY date 
    """)

    rows = db.execute(
        query, 
        {
        "ticker": support, 
        "start_date": start_date,
        "end_date": end_date
        }
    ).mappings().all()

    return rows

def get_rows_between_start_end(
    db: Session,
    ticker: str,
    start_date: date,
    end_date: date,
        
):
    query = text(
        """
        SELECT 
            date
        FROM market_prices
        WHERE ticker = :ticker 
        AND date BETWEEN :start_date AND :end_date;
        """)
    
    rows = db.execute(
        query, 
        {
        "ticker": ticker, 
        "start_date": start_date,
        "end_date": end_date
        }
    ).mappings().all()
    return rows

def get_ticker_window_range(
    db: Session,
    ticker: str,
    end_date: date,
    range: int,
): 
    query = text(
    """
    SELECT 
        ticker,
        date,
        open,
        high,
        low,
        close,
        adjusted_close,
        volume
    FROM market_prices
    WHERE ticker = :ticker AND date <= :end_date
    ORDER BY date DESC
    LIMIT :range;
    """)

    rows = db.execute(
        query, 
        {
        "ticker": ticker, 
        "range": range,
        "end_date": end_date
        }
    ).mappings().all()

    return rows
