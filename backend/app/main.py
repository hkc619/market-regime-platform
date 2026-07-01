from contextlib import asynccontextmanager ## Lifespan of FastAPI, what needs to do when server start and shut down
from fastapi import FastAPI
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')

from core.model_state import ModelState
from core.logging import setup_logging, get_logger

from services.model_registry import load_cnn_gru_model
from ml.model import DualCNNGRUFusion
from middlewares.request_logging import RequestLoggingMiddleware

from api.health import router as health_router
from api.prediction import router as prediction_router
from api.model import router as model_router
from api.support import router as support_router
from api.data import router as data_router
from app.api.prediction_db import router as prediction_db_router

## logging
setup_logging()
logger = get_logger("main")

## lifespan of server
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_state = ModelState() ## construct a global (for all fastapi service) instance of modelstate class

    logger.info("Starting Market Regime Inference API")

    try:
        logger.info("Loading model version=v1 device=cpu")

        loaded = load_cnn_gru_model(version="v2", device="cpu")
        
        app.state.model_state.model_loaded = True
        app.state.model_state.model = loaded["model"]
        app.state.model_state.metadata = loaded["metadata"]
        app.state.model_state.version = loaded["version"]
        app.state.model_state.device = loaded["device"]
        app.state.model_state.error = None

        logger.info(
            "Model loaded successfully | version=%s | device=%s",
            loaded["version"],
            loaded["device"],
        )
        

    except Exception as e:
        app.state.model_state.model_loaded = False
        app.state.model_state.error = str(e)

        logger.exception("Model failed to load")

    yield

    logger.info("Shutting down Market Regime Inference API")

    # shutdown cleanup
    app.state.model_state.model = None
    app.state.model_state.model_loaded = False

app = FastAPI(
    title="Market Regime Inference API",
    version="0.1.1",
    lifespan=lifespan # execute the lifespan when server activating
)

@app.get("/")
async def root():

    logger.info("Hello world")
    return {"message:": "Hello world"}

## middleware
app.add_middleware(RequestLoggingMiddleware)

app.include_router(health_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(support_router, prefix="/api/v1")
app.include_router(data_router, prefix="/api/v1")
app.include_router(prediction_db_router, prefix="/api/v1")