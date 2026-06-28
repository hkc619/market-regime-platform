from fastapi import APIRouter, Request, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from db.session import get_db


router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check(request: Request):
    model_state = request.app.state.model_state
    if model_state.model_loaded:
        return {
            "status": "ok",
            "service": "market-regime-inference-api",
            "model_loaded": True,
            "model_version": model_state.version,
            "device": model_state.device,
        }

    return {
        "status": "degraded",
        "service": "market-regime-inference-api",
        "model_loaded": False,
        "model_version": None,
        "device": model_state.device,
        "error": model_state.error,

    }

@router.get("/health/db")
def database_health_check(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT 1")).scalar()

    return {
        "status": "ok",
        "database": "connected",
        "result": result,
    }