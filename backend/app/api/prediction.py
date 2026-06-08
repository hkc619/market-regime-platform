from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["Prediction"])

class PredictRequest(BaseModel):
    ticker: str

@router.post("/predict")
def predict(request_body: PredictRequest, request: Request):
    model_state = request.app.state.model_state

    if not model_state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": "Model is not loaded. Check /health for details.",
            },
        )

    if request_body.ticker.upper() != "SPY":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_ticker",
                "message": "Model v1 currently supports validated inference only for SPY.",
                "validated_inference_assets": ["SPY"],
                "requested_ticker": request_body.ticker.upper(),
            },
        )

    model = model_state.model
    metadata = model_state.metadata

    result = {
        "ticker": "SPY",
        "model_version": model_state.version,
        "predicted_state": "Trending-Up",
        "confidence": 0.8231,
        "probabilities": {
            "Trending-Down": 0.0245,
            "Trans-Down": 0.0612,
            "Trans-Up": 0.0912,
            "Trending-Up": 0.8231,
        },
    }

    return result