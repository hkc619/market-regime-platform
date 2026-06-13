# backend/app/schemas/prediction.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
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