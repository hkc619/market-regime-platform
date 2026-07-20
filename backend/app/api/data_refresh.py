from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.data_refresh import MarketRefreshRequest, MarketRefreshResult
from app.core.exceptions import ExternalDataFetchError, InvalidExternalDataError
from app.services.data_refresh_service import refresh_market_data
from app.schemas.data_refresh import DataUpdateLogResponse
from app.repository.data_update_log_repository import get_recent_data_update_logs


router = APIRouter(prefix="/data/refresh", tags=["data-refresh"])

@router.post("/market", response_model=MarketRefreshResult)
def refresh_market(
    payload: MarketRefreshRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    request_id = request.state.request_id

    try:
        return refresh_market_data(
            db=db,
            ticker=payload.ticker,
            request_id=request_id,
        )
    except Exception as e:
        raise ExternalDataFetchError(f"Unexpected refresh error: {str(e)}")

    except Exception as e:
        raise InvalidExternalDataError(f"Unexpected refresh error: {str(e)}")
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/logs", response_model=DataUpdateLogResponse)
def get_data_refresh_logs(
    ticker: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    logs = get_recent_data_update_logs(
        db=db,
        ticker=ticker,
        status=status,
        limit=limit,
    )

    return {
        "count": len(logs),
        "logs": logs,
    }