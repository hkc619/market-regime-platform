from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from db.session import get_db


router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status")
def get_market_data_status(
    ticker: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
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

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No market price data found for ticker: {ticker}",
        )

    required_days = 312
    available_days = row["row_count"]

    return {
        "ticker": row["ticker"],
        "available_days": available_days,
        "required_days": required_days,
        "is_ready_for_prediction": available_days >= required_days,
        "start_date": row["start_date"],
        "end_date": row["end_date"],
    }


@router.get("/macro/status")
def get_macro_daily_status(db: Session = Depends(get_db)):
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

    return {
        "total_rows": row["total_rows"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "vix_count": row["vix_count"],
        "yield_10yr_count": row["yield_10yr_count"],
        "yield_2yr_count": row["yield_2yr_count"],
    }


@router.get("/window")
def get_window(
    ticker: str = Query(..., min_length=1),
    lookback: int = Query(..., min=1),
    db: Session = Depends(get_db),
    ):
    
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

    
    available_days = row["row_count"]

    return {
        "ticker": row["ticker"],
        "available_days": available_days,
        "required_days": lookback,
        "is_ready": available_days == lookback,
        "start_date": row["start_date"],
        "end_date": row["end_date"],
    }
    
