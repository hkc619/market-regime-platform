from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel
from pathlib import Path
from sqlalchemy.orm import Session

from app.services.inference_db_service import prepare_latest_inference_input
from app.services.features_input_service import build_latest_model_input
from app.services.inference import predict_proba
from app.repository.prediction_repository import create_prediction_history, get_latest_prediction_by_ticker
from app.core.logging import get_logger
from app.db.session import get_db
from app.core.model_config import RAW_LOOKBACK_DAYS, MODEL_VERSION


logger = get_logger("prediction_db")

router = APIRouter(prefix="/predictions", tags=["Prediction_db"])

class PredictRequest(BaseModel):
    ticker: str
    
@router.post("/latest")
def predict(
    request_body: PredictRequest, 
    request: Request,
    db: Session = Depends(get_db)
    ):
    
    model_state = request.app.state.model_state
    ticker = request_body.ticker.upper()
    
    logger.info(
        "Prediction requested | request_id= | ticker=%s",
        ticker,
    )

    if not model_state.model_loaded:
        logger.warning(
            "Prediction rejected: model not loaded | request_id= | ticker=%s | error=%s",
            ticker,
            model_state.error,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": "Model is not loaded. Check /health for details.",
            },
        )

    if ticker != "SPY":
        logger.warning(
            "Prediction rejected: unsupported ticker | request_id= | ticker=%s",
            ticker,
        )

        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_ticker",
                "message": "Model v1 currently supports validated inference only for SPY.",
                "validated_inference_assets": ["SPY"],
                "requested_ticker": request_body.ticker.upper(),
            },
        )
    
    # if not data_source.exists():
    #     logger.warning(
    #         "Prediction rejected: data_source_not_found | request_id= | ticker=%s",
    #         ticker,
    #     )

    #     raise HTTPException(
    #         status_code=503,
    #         detail={
    #             "error": "data_source_not_found",
    #             "message": "Market data file not found.",
    #             "expected_path": "/Users/hkc619/Documents/PY/project/market-regime-platform/models/metadata.json"
    #         }
    #     )

    '''
    if data_length < 60:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "insufficient_data",
                "message": "At least 60 valid observations are required for prediction.",
                "required_observations": 60,
                "available_observations": 42
            }
        )
    '''
    try:
        model = model_state.model
        metadata = model_state.metadata
        device = model_state.device
        
        raw = prepare_latest_inference_input(
            db=db, 
            ticker=ticker, sup0="QQQ", sup1="TLT", 
            lookback=312
        )

        bundle = build_latest_model_input(raw)

        prediction = predict_proba(bundle, device, model, metadata["model_config"]["scaler_path"])
        print(prediction)
        saved = create_prediction_history(
            db=db,
            ticker=ticker,
            as_of_date=bundle.end_date,
            predicted_class=prediction["predicted_class"],
            predicted_regime=prediction["predicted_regime"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            model_version=MODEL_VERSION,
            raw_window_rows=len(raw.ticker_close),
            feature_rows=bundle.feature_rows,
            model_input_rows=bundle.latest_60_feat.shape[0],
            feature_dim=bundle.feature_dim,
            input_start_date=bundle.start_date,
            input_end_date=bundle.end_date,
        )

        # Prediction completed 
        logger.info(
        "Prediction completed | request_id= | ticker=%s | raw_rows= | feature_rows= | input_shape= | predicted=",
        ticker,
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
            "Prediction failed | request_id= | ticker=%s",
            ticker,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "message": "Prediction failed due to an internal error.",
            },
        ) 
        

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.get("/latest")
def get_latest_predict(
    ticker: str = Query(..., min_length=1),
    db: Session = Depends(get_db)
    ):

    ticker = ticker.upper().strip()

    result = get_latest_prediction_by_ticker(
        db=db, 
        ticker=ticker
    )
    return result


        
    



