from fastapi import APIRouter, Request

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