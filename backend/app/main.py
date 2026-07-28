import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.prediction import router as prediction_router
from app.core.config import get_settings
from app.services.inference import PricePredictor
from app.services.preprocessing import LocationRegistry
from app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + locations list once, before the app starts
    accepting requests (Phase 3, step 5 of the guide)."""
    settings = get_settings()

    predictor = PricePredictor(settings.MODEL_PATH)
    try:
        predictor.load()
    except FileNotFoundError as exc:
        # Don't crash the whole app - let /health report the problem instead
        # of every request failing with an unhelpful stack trace.
        logger.error(str(exc))

    locations = LocationRegistry(settings.LOCATIONS_PATH)

    app.state.predictor = predictor
    app.state.locations = locations

    yield

    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(prediction_router, tags=["prediction"])

    return app


app = create_app()
