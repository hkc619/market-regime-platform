from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.data_refresh import MarketRefreshRequest, MarketRefreshResponse
from app.core.exceptions import ExternalDataFetchError, InvalidExternalDataError
from app.services.data_refresh_service import refresh_market_data


router = APIRouter(prefix="/data/refresh", tags=["data-refresh"])

@router.post("/market", response_model=MarketRefreshResponse)
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
    except ExternalDataFetchError as e:
        raise HTTPException(status_code=502, detail=e.message)

    except InvalidExternalDataError as e:
        raise HTTPException(status_code=502, detail=e.message)
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))