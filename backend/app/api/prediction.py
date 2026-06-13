from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from services.inference_service import load_data, predict_proba
from core.logging import get_logger

logger = get_logger("prediction")

router = APIRouter(tags=["Prediction"])

class PredictRequest(BaseModel):
    ticker: str
    data_source: str

@router.post("/predict")
def predict(request_body: PredictRequest, request: Request):
    
    model_state = request.app.state.model_state
    data_source = Path(request_body.data_source)
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
    
    if not data_source.exists():
        logger.warning(
            "Prediction rejected: data_source_not_found | request_id= | ticker=%s",
            ticker,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "data_source_not_found",
                "message": "Market data file not found.",
                "expected_path": "/Users/hkc619/Documents/PY/project/market-regime-platform/models/metadata.json"
            }
        )

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

        latest_60_feat, latest_regime, idx = load_data(data_source)
        result = predict_proba(latest_60_feat, latest_regime, idx, device, model, metadata["model_config"]["scaler_path"])

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
    
    return result

        




