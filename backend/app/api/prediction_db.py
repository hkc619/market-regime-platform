from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from pathlib import Path
from sqlalchemy.orm import Session

from app.services.inference_db_service import prepare_latest_inference_input
from app.services.features_input_service import build_latest_model_input
from app.services.inference import predict_proba
from app.core.logging import get_logger
from app.db.session import get_db

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
        
        raw = prepare_latest_inference_input(db, ticker, sup0="QQQ", sup1="TLT", lookback=312)
        bundle = build_latest_model_input(raw)
        result = predict_proba(bundle, device, model, metadata["model_config"]["scaler_path"])

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
    
    # Prediction completed | ticker=SPY | raw_rows=332 | feature_rows=80 | input_shape=(60, 35) | predicted=Trending-Up
    logger.info(
        "Prediction completed | request_id= | ticker=%s | raw_rows= | feature_rows= | input_shape= | predicted=",
        ticker,
    )
    return result

        




