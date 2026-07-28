"""
Turns a validated PredictionRequest into the exact one-row DataFrame shape
that the exported sklearn Pipeline expects.

Because the notebook exported a full Pipeline (ColumnTransformer + model),
we do NOT one-hot encode or scale anything here - the pipeline does that
internally. We only need to:
  1. Build a DataFrame with the same column names used at training time.
  2. Map unknown/rare locations to "other", matching the notebook's
     "keep top-N locations" step (Phase 2.3, point 5).
"""
import json
import logging
from pathlib import Path

import pandas as pd

from app.schemas.prediction import PredictionRequest

logger = logging.getLogger(__name__)

# Column names must match numeric_features + categorical_features from the notebook
NUMERIC_FEATURES = ["carpet_area_sqft", "floor_num", "bathroom", "Balcony"]
CATEGORICAL_FEATURES = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]


class LocationRegistry:
    """Loads the allowed-locations list exported by the notebook and
    maps anything outside that list to 'other', mirroring training-time
    behaviour."""

    def __init__(self, locations_path: str):
        self._locations_path = Path(locations_path)
        self._allowed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._locations_path.exists():
            logger.warning(
                "locations.json not found at %s - all locations will be "
                "mapped to 'other'. Copy the file exported by the notebook "
                "into the models/ folder.",
                self._locations_path,
            )
            return
        with self._locations_path.open(encoding="utf-8") as f:
            self._allowed = set(json.load(f))

    def normalize(self, location: str) -> str:
        return location if location in self._allowed else "other"

    @property
    def allowed(self) -> list[str]:
        return sorted(self._allowed)


def build_feature_frame(request: PredictionRequest, locations: LocationRegistry) -> pd.DataFrame:
    """Build the single-row DataFrame handed to model.predict()."""
    row = {
        "carpet_area_sqft": request.carpet_area_sqft,
        "floor_num": request.floor_num,
        "bathroom": request.bathroom,
        "Balcony": request.balcony,
        "location_grouped": locations.normalize(request.location),
        "Furnishing": request.furnishing,
        "Transaction": request.transaction,
        "Ownership": request.ownership,
        "facing": request.facing,
    }
    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
