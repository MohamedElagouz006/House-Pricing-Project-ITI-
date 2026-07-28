"""
Loads the trained pipeline (house_price.pkl) once and exposes a predict()
call used by the /predict route. Loading happens at app startup via the
FastAPI lifespan in main.py - never inside a request handler.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised if /predict is called before the model finished loading."""


class PricePredictor:
    def __init__(self, model_path: str):
        self._model_path = Path(model_path)
        self._model = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self._model_path}'. "
                "Export house_price.pkl from the notebook (Phase 2.6) and "
                "copy it into backend/models/."
            )
        logger.info("Loading model from %s", self._model_path)
        self._model = joblib.load(self._model_path)
        logger.info("Model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, features: pd.DataFrame) -> float:
        if self._model is None:
            raise ModelNotLoadedError("Model is not loaded yet.")
        prediction = self._model.predict(features)
        return float(prediction[0])
