from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.services.inference import ModelNotLoadedError
from app.services.preprocessing import build_feature_frame

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    return HealthResponse(status="ok", model_loaded=predictor.is_loaded)


@router.get("/locations", response_model=list[str])
def get_locations(request: Request) -> list[str]:
    """Returns the list of known locations so the frontend can populate
    its dropdown (Phase 4, requirement 2)."""
    return request.app.state.locations.allowed


@router.post("/predict", response_model=PredictionResponse)
def predict_price(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor = request.app.state.predictor
    locations = request.app.state.locations

    try:
        features = build_feature_frame(payload, locations)
        predicted_price = predictor.predict(features)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Try again in a moment.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean 500 for the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    if predicted_price < 0:
        predicted_price = 0.0

    return PredictionResponse(predicted_price=round(predicted_price, 2))
