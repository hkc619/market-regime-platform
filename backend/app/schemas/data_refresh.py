# app/schemas/data_refresh.py

from datetime import date
from pydantic import BaseModel, Field


class MarketRefreshRequest(BaseModel):
    ticker: str
    # tickers: list[str] = Field(default=["SPY", "QQQ", "TLT"])


class MarketRefreshResult(BaseModel):
    ticker: str
    latest_before: date | None
    latest_after: date | None
    rows_fetched: int
    rows_inserted_or_updated: int
    status: str


class MarketRefreshResponse(BaseModel):
    status: str
    results: MarketRefreshResult