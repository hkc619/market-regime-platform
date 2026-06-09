from fastapi import APIRouter, Request

router = APIRouter(tags=["Support"])

@router.get("/supported-assets")
def supported_assets(request: Request):
    model_state = request.app.state.model_state
    if model_state.model_loaded:
        metadata = model_state.metadata
        return {
            "validated_inference_assets": metadata["assets"],
            "available_data_assets": metadata["data_sources"]["price_sheets", "macro_daily", "macro_monthly"],
            "experimental_inference_assets": [],
            "default_asset": metadata["assets"],
            "note": "Model v1 is trained and validated for SPY. The preprocessing pipeline is designed to be ticker-agnostic, but non-SPY inference is not enabled in this version."
        }

    return {
        'Message': "Supported assets info not found."
    }