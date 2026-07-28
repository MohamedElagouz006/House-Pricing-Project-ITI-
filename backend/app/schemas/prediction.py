"""
Request/response models for the /predict endpoint.

IMPORTANT: these fields must match the features your notebook trained on
(Phase 2.4 in the project guide). If you added/removed/renamed a feature
in the notebook, mirror that change here AND in services/preprocessing.py.
"""
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    location: str = Field(..., min_length=1, examples=["Sector 45"])
    carpet_area_sqft: float = Field(..., gt=0, examples=[1200.0])
    floor_num: int = Field(..., ge=0, examples=[3])
    bathroom: int = Field(..., ge=0, examples=[2])
    balcony: int = Field(..., ge=0, examples=[1])
    furnishing: str = Field(..., examples=["Semi-Furnished"])
    transaction: str = Field(..., examples=["Resale"])
    ownership: str = Field(..., examples=["Freehold"])
    facing: str = Field(..., examples=["East"])

    @field_validator("location", "furnishing", "transaction", "ownership", "facing")
    @classmethod
    def strip_and_require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR"


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
