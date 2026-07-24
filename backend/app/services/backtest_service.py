import time
from decimal import Decimal
from typing import Any
from datetime import date

from app.core.logging import get_logger
from app.core.model_config import RAW_LOOKBACK_DAYS, MODEL_VERSION

logger = get_logger(__name__)

def predict_for_date(
        db,
        ticker,
        as_of_date: date,
        model_state,
        request_id,
) -> dict:
    """
    Run model inference using data available up to as_of_date.

    persist=False is used by backtesting so historical predictions
    are not written into the production prediction history table.
    """
    
    from app.services.inference_db_service import prepare_inference_input
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
        raw = prepare_inference_input(
            db=db, 
            ticker=ticker, sup0="QQQ", sup1="TLT",
            latest=False, 
            lookback=312
        )

        raw_rows = len(raw.ticker_close)

        logger.info("Raw inference loaded | request_id=%s | ticker=%s | raw_rows=%d ",
            request_id,
            ticker,
            raw_rows,
        )
        if raw_rows == 0:
            raise TickerNotFoundError(f"No market data found for ticker={ticker}")
        
        if raw_rows < RAW_LOOKBACK_DAYS:
            raise InsufficientRawDataError(
                f"{ticker} only has {raw_rows} raw rows. "
                f"Need at least {RAW_LOOKBACK_DAYS}"
            )

        model_input = build_latest_model_input(raw)

        logger.info(
            "Model input built | request_id=%s | ticker=%s | feature_rows=%s | input_shape=%s",
            request_id,
            ticker,
            model_input.feature_rows,
            model_input.latest_60_feat.shape,
        )

        if model_input.latest_60_feat.shape[0] < 60:
            raise InsufficientFeatureDataError(
                f"{ticker} does not have enough valid feature rows for inference."
        )

        try:
            prediction = predict_proba(model_input, device, model, metadata["model_config"]["scaler_path"])
        except Exception as e:
            logger.exception(
                "Model inference failed | request_id=%s | ticker=%s",
                request_id,
                ticker,
            )
            raise ModelInferenceError(str(e)) from e
        
        logger.info(
           "Prediction completed | request_id=%s | ticker=%s | regime=%s | confidence=%.4f",
            request_id,
            ticker,
            prediction["predicted_regime"],
            prediction["confidence"],
        )

        try:
            saved = create_prediction_history(
                db=db,
                ticker=ticker,
                as_of_date=model_input.end_date,
                predicted_class=prediction["predicted_class"],
                predicted_regime=prediction["predicted_regime"],
                confidence=prediction["confidence"],
                probabilities=prediction["probabilities"],
                model_version=MODEL_VERSION,
                raw_window_rows=len(raw.ticker_close),
                feature_rows=model_input.feature_rows,
                model_input_rows=model_input.latest_60_feat.shape[0],
                feature_dim=model_input.feature_dim,
                input_start_date=model_input.start_date,
                input_end_date=model_input.end_date,
        )
        except Exception as e:
            logger.exception(
                "Prediction save failed | request_id=%s | ticker=%s",
                request_id,
                ticker,
            )
            raise PredictionSaveError(str(e)) from e
        
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Prediction request finished | request_id=%s | ticker=%s | prediction_id=%s | latency_ms=%s",
            request_id,
            ticker,
            saved["id"],
            latency_ms,
        )


        return {
                "prediction_id": saved["id"],
                "ticker": saved["ticker"],
                "as_of_date": saved["as_of_date"],
                "predicted_class": saved["predicted_class"],
                "predicted_regime": saved["predicted_regime"],
                "confidence": float(saved["confidence"]),
                "probabilities": {
                    "Trending-Down": float(saved["prob_trending_down"]),
                    "Transition-Down": float(saved["prob_transition_down"]),
                    "Transition-Up": float(saved["prob_transition_up"]),
                    "Trending-Up": float(saved["prob_trending_up"]),
                },
                "model_version": saved["model_version"],
                "input_window": {
                    "raw_window_rows": saved["raw_window_rows"],
                    "feature_rows": saved["feature_rows"],
                    "model_input_rows": saved["model_input_rows"],
                    "feature_dim": saved["feature_dim"],
                    "input_start_date": saved["input_start_date"],
                    "input_end_date": saved["input_end_date"],
                },
                "created_at": saved["created_at"],
            }
    except Exception:
        logger.exception(
            "Prediction request failed | request_id=%s | ticker=%s",
            request_id,
            ticker,
        )
        raise
