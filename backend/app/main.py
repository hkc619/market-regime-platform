from contextlib import asynccontextmanager ## Lifespan of FastAPI, what needs to do when server start and shut down

from fastapi import FastAPI

from core.model_state import ModelState
from services.model_registry import load_cnn_gru_model
from ml.model import DualCNNGRUFusion
from api.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_state = ModelState() ## construct a global (for all fastapi service) instance of modelstate class

    try:
        loaded = load_cnn_gru_model(version="v1", device="cpu")

        app.state.model_state.model_loaded = True
        app.state.model_state.model: loaded["model"]
        app.state.model_state.metadata: loaded["metadata"]
        app.state.model_state.version: loaded["version"]
        app.state.model_state.device: loaded["device"]
        app.state.model_state.error: None

        print("Model loaded successfully")

    except Exception as e:
        app.state.model_state.model_loaded = False
        app.state.model_state.error = str(e)

        print(f"Model failed to load: {e}")

    yield

    app.state.model_state.model = None
    app.state.model_state.model_loaded = False

app = FastAPI(
    title="Market Regime Inference API",
    version="0.1.0",
    lifespan=lifespan # execute the lifespan when server activating
)

@app.get("/")
async def root():
    return {"message": "Hello world."}

## middleware

app.include_router(health_router)