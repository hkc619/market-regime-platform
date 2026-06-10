# backend/app/schemas/common.py

from pydantic import BaseModel
from typing import Optional, Any


class ErrorDetail(BaseModel):
    error: str
    message: str
    requested_ticker: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_version: Optional[str] = None
    device: str = "cpu"
    error: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    message: str


class APIErrorResponse(BaseModel):
    detail: ErrorDetail