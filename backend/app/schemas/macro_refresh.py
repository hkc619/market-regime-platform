from datetime import date, datetime
from pydantic import BaseModel, Field


class MacroRefreshRequest(BaseModel):
    type: str = Field(..., min_length=1, examples="daily")


class MacroRefreshResult(BaseModel):
    type_of_macro: str 
    latest_before: date | None
    latest_after: date | None
    rows_fetched: int
    rows_inserted_or_updated: int
    status: str
    message: str