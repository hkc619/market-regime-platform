from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.data_refresh import MarketRefreshRequest, MarketRefreshResponse
from app.services.data_refresh_service import refresh_market_data, get_latest_market_date


router = APIRouter(prefix="/data/refresh", tags=["data-refresh"])

# response_model=MarketRefreshResponse

@router.post("/market")
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

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))