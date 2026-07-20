from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.services.prediction_service import create_latest_prediction

from app.repository.prediction_repository import get_latest_prediction_by_ticker

from app.core.exceptions import (
    AppError,
    InsufficientFeatureDataError,
    InsufficientRawDataError,
    ModelInferenceError,
    PredictionSaveError,
    TickerNotFoundError,
)
from app.core.logging import get_logger

from app.db.session import get_db

from app.schemas.prediction import (
    LatestPredictionRequest,
    LatestPredictionResponse,
)

logger = get_logger("prediction_db")

router = APIRouter(prefix="/predictions", tags=["Prediction_db"])
    
@router.post("/latest", response_model=LatestPredictionResponse)
def predict(
        request_body: LatestPredictionRequest, 
        request: Request,
        db: Session = Depends(get_db)
    ):
    
    model_state = request.app.state.model_state
    request_id = request.state.request_id
    ticker = request_body.ticker.upper()
    

    if not model_state.model_loaded:
        logger.warning(
            "Prediction rejected: model not loaded | request_id=%s | ticker=%s | error=%s",
            request_id,
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
            "Prediction rejected: unsupported ticker | request_id=%s | ticker=%s",
            request_id,
            ticker,
        )

        raise HTTPException(
            status_code=404,
            detail={
                "error": "unsupported_ticker",
                "message": "Model v1 currently supports validated inference only for SPY.",
                "validated_inference_assets": ["SPY"],
                "requested_ticker": request_body.ticker.upper(),
            },
        )
    try:
        return create_latest_prediction(
            db=db,
            ticker=ticker,
            model_state=model_state, 
            request_id=request_id,
        )

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
        
    except AppError:
        raise

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        raise TickerNotFoundError(f"Unexpected prediction error: {str(e)}")

    except Exception as e:
        raise InsufficientRawDataError(f"Unexpected prediction error: {str(e)}")
    
    except Exception as e:
        raise InsufficientFeatureDataError(f"Unexpected prediction error: {str(e)}")

    except Exception as e:
        raise ModelInferenceError(f"Model inference failed: {str(e)}")

    except Exception as e:
        raise PredictionSaveError(f"Prediction save failed: {str(e)}")
    
    

@router.get("/latest", response_model=LatestPredictionResponse)
def get_latest_predict(
    ticker: str = Query(..., min_length=1),
    db: Session = Depends(get_db)
    ):

    ticker = ticker.upper().strip()

    result = get_latest_prediction_by_ticker(
        db=db, 
        ticker=ticker
    )
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction history found for ticker={ticker.upper()}",
        )

    return {
        "prediction_id": result["id"],
        "ticker": result["ticker"],
        "as_of_date": result["as_of_date"],
        "predicted_class": result["predicted_class"],
        "predicted_regime": result["predicted_regime"],
        "confidence": float(result["confidence"]),
        "probabilities": {
            "Trending-Down": float(result["prob_trending_down"]),
            "Transition-Down": float(result["prob_transition_down"]),
            "Transition-Up": float(result["prob_transition_up"]),
            "Trending-Up": float(result["prob_trending_up"]),
        },
        "model_version": result["model_version"],
        "input_window": {
            "raw_window_rows": result["raw_window_rows"],
            "feature_rows": result["feature_rows"],
            "model_input_rows": result["model_input_rows"],
            "feature_dim": result["feature_dim"],
            "input_start_date": result["input_start_date"],
            "input_end_date": result["input_end_date"],
        },
        "created_at": result["created_at"],
    }


        
    



