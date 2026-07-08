# app/schemas/data_refresh.py

from datetime import date, datetime
from pydantic import BaseModel, Field


class MarketRefreshRequest(BaseModel):
    ticker: str = Field(..., min_length=1, examples=["SPY"])


class MarketRefreshResult(BaseModel):
    ticker: str
    latest_before: date | None
    latest_after: date | None
    rows_fetched: int
    rows_inserted_or_updated: int
    status: str
    message: str


class MarketRefreshResponse(BaseModel):
    status: str
    results: MarketRefreshResult


class DataUpdateLogItem(BaseModel):
    id: int
    request_id: str | None
    data_type: str
    source: str
    ticker: str | None
    start_date: date | None
    end_date: date | None
    status: str
    rows_fetched: int
    rows_inserted_or_updated: int
    error_message: str | None
    created_at: datetime


class DataUpdateLogResponse(BaseModel):
    count: int
    logs: list[DataUpdateLogItem]