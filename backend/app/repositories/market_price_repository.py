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

def get_ticker_window_ending_at(
    db: Session,
    ticker: str,
    as_of_date: date,
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
    WHERE ticker = :ticker AND date <= :as_of_date
    ORDER BY date DESC
    LIMIT :lookback;
    """)

    rows = db.execute(
        query, 
        {
        "ticker": ticker, 
        "lookback": lookback,
        "as_of_date": as_of_date
        }
    ).mappings().all()

    return rows
