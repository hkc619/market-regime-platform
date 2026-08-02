import time
from decimal import Decimal
from typing import Any
from datetime import date

from app.schemas.backtest import (
    BacktestPredictionItem, 
    ConfidenceSummary, 
    BacktestSummary
    )
from app.db.session import get_db
from app.core.logging import get_logger
from app.core.model_config import RAW_LOOKBACK_DAYS, MODEL_VERSION
from app.core.exceptions import (
    InsufficientFeatureDataError,
    InsufficientRawDataError,
    ModelInferenceError,
    TickerNotFoundError,
    NoSuccessfulBacktestPredictionsError,
    InvalidPredictionError,
)

from app.services.data_service import (
    get_latest_support_window, 
    get_macro_daily_window, 
    get_macro_monthly_window,
    )

from app.services.backtest_data_service import get_range_ticker_prices, get_rows_between_start_end

logger = get_logger(__name__)

def predict_for_date(
        ticker,
        model_state,
        request_id,
        ticker_rows,
        sup0_rows,
        sup1_rows,
        macro_daily_rows,
        macro_monthly_rows
) :
    """
    Run model inference using data available up to as_of_date.

    persist=False is used by backtesting so historical predictions
    are not written into the production prediction history table.
    """
    
    from app.services.inference_db_service import prepare_backtest_inference_input
    from app.services.inference import predict_proba
    from app.services.features_input_service import build_latest_model_input

    start_time = time.perf_counter()
    ticker = ticker.upper().strip()

    logger.info(
        "Prediction request started | request_id=%s | ticker=%s",
        request_id,
        ticker,   
    )

    model = model_state.model
    metadata = model_state.metadata
    device = model_state.device
    try: 
        raw = prepare_backtest_inference_input(
            ticker=ticker,
            ticker_rows=ticker_rows, 
            sup0_rows=sup0_rows, 
            sup1_rows=sup1_rows,
            macro_daily_rows=macro_daily_rows,
            macro_monthly_rows=macro_monthly_rows
        )

        raw_rows = len(raw.ticker_close)
        
        if raw_rows == 0:
            raise TickerNotFoundError(f"No market data found for ticker={ticker}")
        
        if raw_rows < RAW_LOOKBACK_DAYS:
            raise InsufficientRawDataError(
                f"{ticker} only has {raw_rows} raw rows. "
                f"Need at least {RAW_LOOKBACK_DAYS}"
            )
        
        model_input = build_latest_model_input(raw)

        if model_input.latest_60_feat.shape[0] < 60:
            raise InsufficientFeatureDataError(
                f"{ticker} does not have enough valid feature rows for inference."
        )
        try:
            prediction = predict_proba(model_input, device, model, metadata["model_config"]["scaler_path"])
        except Exception as e:
            raise ModelInferenceError(str(e)) from e

        return {
            "ticker":ticker,
            "as_of_date": model_input.end_date,
            "predicted_class":int(prediction["predicted_class"]),
            "predicted_regime":prediction["predicted_regime"],
            "confidence":float(prediction["confidence"]),
            "probabilities": prediction["probabilities"],
        }
    except:
        pass
        

STATE_NAMES = {
    0: "Trending-Down",
    1: "Transition-Down",
    2: "Transition-Up",
    3: "Trending-Up",
}

def build_backtest_summary(
    *,
    ticker: str,
    requested_start_date: date,
    requested_end_date: date,
    candidate_dates: list[date],
    predictions: list[BacktestPredictionItem],
    skip_reasons: dict[str, int],
) -> BacktestSummary:
    if not predictions:
        raise NoSuccessfulBacktestPredictionsError(
            "Backtest produced no successful predictions."
        )

    regime_distribution = {
        regime_name: 0
        for regime_name in STATE_NAMES.values()
    }

    confidences: list[float] = []

    for prediction in predictions:
        expected_regime = STATE_NAMES.get(
            prediction.predicted_class
        )

        if expected_regime is None:
            raise InvalidPredictionError(
                "Unknown predicted class: "
                f"{prediction.predicted_class}"
            )

        if prediction.predicted_regime != expected_regime:
            raise InvalidPredictionError(
                "Predicted class and regime do not match: "
                f"class={prediction.predicted_class}, "
                f"expected={expected_regime}, "
                f"actual={prediction.predicted_regime}"
            )

        regime_distribution[
            prediction.predicted_regime
        ] += 1

        confidences.append(prediction.confidence)

    num_candidate_dates = len(candidate_dates)
    num_predictions = len(predictions)
    skipped_dates = num_candidate_dates - num_predictions

    coverage_rate = (
        num_predictions / num_candidate_dates
        if num_candidate_dates > 0
        else 0.0
    )

    regime_distribution_pct = {
        regime_name: round(
            count / num_predictions,
            4,
        )
        for regime_name, count
        in regime_distribution.items()
    }

    confidence_summary = ConfidenceSummary(
        average=round(
            sum(confidences) / len(confidences),
            4,
        ),
        minimum=round(min(confidences), 4),
        maximum=round(max(confidences), 4),
        low_confidence_count=sum(
            confidence < 0.5
            for confidence in confidences
        ),
    )

    return BacktestSummary(
        ticker=ticker,
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        actual_prediction_start_date=predictions[
            0
        ].as_of_date,
        actual_prediction_end_date=predictions[
            -1
        ].as_of_date,
        num_candidate_dates=num_candidate_dates,
        num_predictions=num_predictions,
        skipped_dates=skipped_dates,
        coverage_rate=round(coverage_rate, 4),
        skip_reasons=skip_reasons,
        regime_distribution=regime_distribution,
        regime_distribution_pct=(
            regime_distribution_pct
        ),
        confidence_summary=confidence_summary,
        predictions=predictions,
    )
        


def backtest_for_range(
        model_state,
        db,
        ticker,
        sup0,
        sup1,
        start_date: date,
        end_date: date,
        request_id
):
    ticker_rows = get_range_ticker_prices(
        db=db,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback=312
    )
    sup0_rows = get_latest_support_window(
        db=db,
        support=sup0,
        start_date=start_date,
        end_date=end_date,

    )
    sup1_rows = get_latest_support_window(
        db=db,
        support=sup1,
        start_date=start_date,
        end_date=end_date,
    )
    macro_daily_rows = get_macro_daily_window(
        db=db
    )
    macro_monthly_rows = get_macro_monthly_window(
        db=db
    )

    candidate_date_rows = get_rows_between_start_end(
        db=db,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    skip_reasons = {
    "insufficient_market_rows": 0,
    "candidate_date_missing": 0,
    "prediction_failed": 0,
}
    
    predictions: list[BacktestPredictionItem] = []

    for i in range(len(candidate_date_rows)):
        start = i
        end = 312 + i + 1
        prediction = predict_for_date(
            ticker,
            model_state=model_state,
            request_id=request_id,
            ticker_rows=ticker_rows[start:end],
            sup0_rows=sup0_rows[start:end],
            sup1_rows=sup1_rows[start:end],
            macro_daily_rows=macro_daily_rows,
            macro_monthly_rows=macro_monthly_rows
        )

        print(prediction)
        prediction_item = BacktestPredictionItem(
            as_of_date=prediction["as_of_date"],
            predicted_class=int(prediction["predicted_class"]),
            predicted_regime=prediction["predicted_regime"],
            confidence=float(prediction["confidence"]),
            probabilities={
                state: float(probability)
                for state, probability in prediction["probabilities"].items()
            },
        )
        predictions.append(prediction_item)
        
        candidate_dates = [
            row["date"]
            for row in candidate_date_rows
        ]

        summary = build_backtest_summary(
            ticker=ticker,
            requested_start_date=start_date,
            requested_end_date=end_date,
            candidate_dates=candidate_dates,
            predictions=predictions,
            skip_reasons=skip_reasons,
        )

    return summary

