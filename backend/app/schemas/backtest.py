from datetime import date

from pydantic import BaseModel, Field


class BacktestPredictionItem(BaseModel):
    as_of_date: date
    predicted_class: int
    predicted_regime: str
    confidence: float
    probabilities: dict[str, float]

class ConfidenceSummary(BaseModel):
    average: float
    minimum: float
    maximum: float
    low_confidence_count: int


class BacktestSummary(BaseModel):
    ticker: str

    requested_start_date: date
    requested_end_date: date

    actual_prediction_start_date: date
    actual_prediction_end_date: date

    num_candidate_dates: int
    num_predictions: int
    skipped_dates: int
    coverage_rate: float

    skip_reasons: dict[str, int]

    regime_distribution: dict[str, int]
    regime_distribution_pct: dict[str, float]

    confidence_summary: ConfidenceSummary


    predictions: list[BacktestPredictionItem] = Field(
        default_factory=list
    )