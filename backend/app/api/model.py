from fastapi import APIRouter, Request

router = APIRouter(tags=["Info"])

@router.get("/model-info")
def health_check(request: Request):
    model_state = request.app.state.model_state
    if model_state.model_loaded:
        metadata = model_state.metadata
        return {
            "model_version": model_state.version,
            "model_name": metadata['model_name'],
            "task": metadata["task"],
            "number of classes":metadata["num_classes"],
            "classes": metadata["classes"],
             "input_config": metadata["input_config"],
            "model_config": metadata["model_config"],
       "metrics": {
  "overall": {
    "accuracy": metadata["metrics"]["overall"]["accuracy"],
    "macro_f1": metadata["metrics"]["overall"]["macro_f1"],
    "total_support": metadata["metrics"]["overall"]["total_support"]}
       }
        }

    return {
        'model': "Model info not found."
    }