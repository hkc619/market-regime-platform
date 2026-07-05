from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repository.data_repository import get_market_data, get_macro_daily, get_window


router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status")
def get_market_data_status(
    ticker: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    
    row = get_market_data(ticker, db)

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
    
    row = get_macro_daily(db=db)

    return {
        "total_rows": row["total_rows"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "vix_count": row["vix_count"],
        "yield_10yr_count": row["yield_10yr_count"],
        "yield_2yr_count": row["yield_2yr_count"],
    }


@router.get("/window")
def get_prediction_window(
    ticker: str = Query(..., min_length=1),
    lookback: int = Query(..., min=1),
    db: Session = Depends(get_db),
    ):
    
    row = get_window(db=db, lookback=lookback, ticker=ticker)
    
    available_days = row["row_count"]

    return {
        "ticker": row["ticker"],
        "available_days": available_days,
        "required_days": lookback,
        "is_ready": available_days == lookback,
        "start_date": row["start_date"],
        "end_date": row["end_date"],
    }
    
