from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import RawTrainingData

def load_market_prices(
    db: Session,
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    query = """
        SELECT
            date,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume
        FROM market_prices
        WHERE ticker = :ticker
    """

    params = {"ticker": ticker}

    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date

    if end_date:
        query += " AND date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY date ASC"

    df = pd.read_sql_query(
        text(query),
        db.bind,
        params=params,
        parse_dates=["date"],
    )

    if df.empty:
        raise ValueError(f"No market price data found for ticker={ticker}")

    df = df.set_index("date").sort_index()

    return df


def load_macro_daily(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    query = """
        SELECT
            date,
            vix,
            yr10,
            yr2
        FROM macro_daily
        WHERE 1 = 1
    """

    params = {}

    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date

    if end_date:
        query += " AND date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY date ASC"

    df = pd.read_sql_query(
        text(query),
        db.bind,
        params=params,
        parse_dates=["date"],
    )

    if df.empty:
        raise ValueError("No macro_daily data found")

    df = df.set_index("date").sort_index()

    return df


def load_macro_monthly(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    query = """
        SELECT
            date,
            cpi_yoy
        FROM macro_monthly
        WHERE 1 = 1
    """

    params = {}

    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date

    if end_date:
        query += " AND date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY date ASC"

    df = pd.read_sql_query(
        text(query),
        db.bind,
        params=params,
        parse_dates=["date"],
    )

    if df.empty:
        raise ValueError("No macro_monthly data found")

    df = df.set_index("date").sort_index()

    return df


def load_training_data_from_db(
    db: Session,
    target_ticker: str = "SPY",
    support1: str = "QQQ",
    support2: str = "TLT",
    start_date: date | None = None,
    end_date: date | None = None,
    
) -> RawTrainingData:
    ticker_prices = load_market_prices(
        db=db,
        ticker=target_ticker,
        start_date=start_date,
        end_date=end_date,
    )


    support1_prices = load_market_prices(
        db=db,
        ticker=support1,
        start_date=start_date,
        end_date=end_date,
    )

    support2_prices = load_market_prices(
    db=db,
    ticker=support2,
    start_date=start_date,
    end_date=end_date,
    )

    macro_daily = load_macro_daily(
        db=db,
        start_date=start_date,
        end_date=end_date,
    )

    macro_monthly = load_macro_monthly(
        db=db,
        start_date=start_date,
        end_date=end_date,
    )

    return RawTrainingData(
        target_prices=ticker_prices,
        sup1_close=support1_prices["close"].rename("Sup1_Close"),
        sup2_close=support2_prices["close"].rename("Sup2_Close"),
        macro_daily=macro_daily,
        macro_monthly=macro_monthly,
    )