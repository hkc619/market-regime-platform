from fastapi import APIRouter, Depends, HTTPException, Request

from sqlalchemy.orm import Session

from app.db.session import get_db

from app.services.macro_refresh_service import MacroDataService
from app.schemas.macro_refresh import  MacroRefreshResult
from app.providers.fred_provider import FredClient
from app.core.exceptions import ExternalDataFetchError, InvalidExternalDataError

router = APIRouter(prefix="/data/refresh", tags=["data-refresh"])

def get_macro_data_service() -> MacroDataService:
    fred_client = FredClient()
    return MacroDataService(fred_client=fred_client)

@router.post("/macro/daily", response_model=MacroRefreshResult)
def refresh_macro_daily(
    request: Request,
    db: Session = Depends(get_db),
    service: MacroDataService = Depends(get_macro_data_service),
):
    request_id = request.state.request_id

    try:
        
        return service.refresh_daily_macro(
            db=db,
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
    
@router.post("/macro/monthly", response_model=MacroRefreshResult)
def refresh_macro_daily(
    request: Request,
    db: Session = Depends(get_db),
    service: MacroDataService = Depends(get_macro_data_service),
):
    request_id = request.state.request_id

    try:
        return service.refresh_monthly_macro(
            db=db,
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
    