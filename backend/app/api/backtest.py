from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import date

from app.services.backtest_service import predict_for_date

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

logger = get_logger("backtest")

router = APIRouter(prefix="/backtest", tags=["backtest"])

class payload(BaseModel):
    ticker : str
    as_of_date: str
    

@router.post("/test")
def backtest(
        request_body : payload, 
        request: Request,
        db: Session = Depends(get_db),
    ):

    model_state = request.app.state.model_state
    request_id = request.state.request_id
    ticker = request_body.ticker.upper()
    as_of_date = date.fromisoformat(request_body.as_of_date)

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
        rows =  predict_for_date(
            db=db,
            ticker=ticker,
            model_state=model_state, 
            request_id=request_id,
            as_of_date=as_of_date
        )
        print(rows)
        return rows


    except Exception:
        logger.exception(
            "Prediction failed | request_id=%s | ticker=%s",
            request_id,
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
    
    

