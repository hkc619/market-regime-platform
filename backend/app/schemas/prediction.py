from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import date, datetime


class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="SPY")


class InputSummary(BaseModel):
    window_size: int
    num_features_per_timestep: int
    num_observations_used: int


class PredictResponse(BaseModel):
    ticker: str
    model_version: str
    latest_observation_date: str
    prediction_target: str
    predicted_state: str
    confidence: float
    probabilities: Dict[str, float]
    input_summary: InputSummary
    warnings: List[str] = []


class BatchPredictRequest(BaseModel):
    tickers: List[str] = Field(..., example=["SPY", "QQQ", "TLT"])


class BatchPredictionItem(BaseModel):
    ticker: str
    status: str
    predicted_state: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    reason: Optional[str] = None
    message: Optional[str] = None


class BatchPredictSummary(BaseModel):
    requested: int
    succeeded: int
    skipped: int
    failed: int


class BatchPredictResponse(BaseModel):
    model_version: str
    results: List[BatchPredictionItem]
    summary: BatchPredictSummary

class LatestPredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, examples=["SPY"])


class PredictionInputWindow(BaseModel):
    raw_window_rows: int
    feature_rows: int
    model_input_rows: int
    feature_dim: int
    input_start_date: date
    input_end_date: date


class LatestPredictionResponse(BaseModel):
    prediction_id: int

    ticker: str
    as_of_date: date

    predicted_class: int
    predicted_regime: str
    confidence: float

    probabilities: dict[str, float]

    model_version: str
    input_window: PredictionInputWindow

    created_at: datetime

class PredictionHistoryItem(BaseModel):
    prediction_id: int
    ticker: str
    as_of_date: date

    predicted_class: int
    predicted_regime: str
    confidence: float = Field(..., ge=0, le=1)

    probabilities: dict[str, float]

    model_version: Optional[str] = None
    created_at: Optional[datetime] = None


class PredictionHistoryResponse(BaseModel):
    ticker: str
    count: int
    results: list[PredictionHistoryItem]